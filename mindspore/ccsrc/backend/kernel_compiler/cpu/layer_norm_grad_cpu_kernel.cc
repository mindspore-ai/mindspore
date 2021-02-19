/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/cpu/layer_norm_grad_cpu_kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
void LayerNormGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  std::vector<size_t> x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto begin_norm_axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "begin_norm_axis");
  auto begin_params_axis = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "begin_params_axis");
  if (begin_norm_axis < 0) {
    begin_norm_axis += x_shape.size();
  }
  if (begin_params_axis < 0) {
    begin_params_axis += x_shape.size();
  }
  for (size_t i = 0; i < IntToSize(begin_norm_axis); i++) {
    block_num_ *= x_shape[i];
  }
  for (size_t i = IntToSize(begin_norm_axis); i < x_shape.size(); i++) {
    block_size_ *= x_shape[i];
  }
  for (size_t i = 0; i < IntToSize(begin_params_axis); i++) {
    param_size_ *= x_shape[i];
  }
  for (size_t i = begin_params_axis; i < x_shape.size(); i++) {
    param_num_ *= x_shape[i];
  }
  if (block_num_ <= 0 || block_size_ <= 0) {
    MS_LOG(EXCEPTION) << "LayerNormGradCPUKernel input shape error, input shape: " << x_shape;
  }
}

bool LayerNormGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, workspace, outputs);
  } else {
    MS_LOG(EXCEPTION) << "input dtype only support float16, float32, float64";
  }
  return true;
}

template <typename T>
void LayerNormGradCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto dy = reinterpret_cast<T *>(inputs[1]->addr);
  auto var = reinterpret_cast<T *>(inputs[2]->addr);
  auto mean = reinterpret_cast<T *>(inputs[3]->addr);
  auto gamma = reinterpret_cast<T *>(inputs[4]->addr);
  auto dx = reinterpret_cast<T *>(outputs[0]->addr);
  auto dg = reinterpret_cast<T *>(outputs[1]->addr);
  auto db = reinterpret_cast<T *>(outputs[2]->addr);
  size_t thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  auto thread_num1 = param_num_ < thread_num ? param_num_ : thread_num;
  std::vector<common::Task> tasks1;
  tasks1.reserve(thread_num1);
  auto thread_num2 = block_num_ < thread_num ? block_num_ : thread_num;
  std::vector<common::Task> tasks2;
  tasks2.reserve(thread_num2);
  auto task1 = [&](size_t start, size_t end) {
    for (size_t c = 0; c < ceil(static_cast<double>(param_num_) / thread_num1); ++c) {
      if (c * thread_num1 + start >= param_num_) {
        continue;
      }
      size_t param_index = c * thread_num1 + start;
      T dgamma = (T)0.0;
      T dbeta = (T)0.0;
      for (size_t j = param_index; j < param_size_ * param_num_; j += param_num_) {
        auto norm_shift = static_cast<int>(j / block_size_);
        dgamma += dy[j] * (T)std::pow(static_cast<double>(var[norm_shift]) + eps_, -0.5) * (x[j] - mean[norm_shift]);
        dbeta += dy[j];
      }
      dg[param_index] = dgamma;
      db[param_index] = dbeta;
    }
  };
  auto task2 = [&](size_t start, size_t end) {
    for (size_t c = 0; c < ceil(static_cast<double>(block_num_) / thread_num2); ++c) {
      if (c * thread_num2 + start >= block_num_) {
        continue;
      }
      size_t block_index = c * thread_num2 + start;
      T sum1 = (T)0.0;
      T sum2 = (T)0.0;
      T sum3 = (T)0.0;
      for (size_t j = block_index * block_size_; j < (block_index + 1) * block_size_; ++j) {
        auto param_shift = j % param_num_;
        auto norm_shift = static_cast<int>(j / block_size_);
        auto dxm = x[j] - mean[norm_shift];
        auto dyg = dy[j] * gamma[param_shift];
        sum1 += (T)(-0.5) * dyg * dxm * (T)std::pow(static_cast<double>(var[norm_shift]) + eps_, -1.5);
        sum2 += dyg;
        sum3 += (T)(-2.0) * dxm;
      }
      for (size_t j = block_index * block_size_; j < (block_index + 1) * block_size_; ++j) {
        auto param_shift = j % param_num_;
        auto norm_shift = static_cast<int>(j / block_size_);
        auto var_sqrt = (T)std::pow(static_cast<double>(var[norm_shift]) + eps_, -0.5);
        auto dx1 = dy[j] * gamma[param_shift] * var_sqrt;
        auto dx2 = sum1 * (T)2.0 / block_size_ * (x[j] - mean[norm_shift]);
        auto dx3 = ((T)(-1.0) * var_sqrt * sum2 + ((T)1.0 / block_size_) * sum1 * sum3) * ((T)1.0 / block_size_);
        dx[j] = dx1 + dx2 + dx3;
      }
    }
  };
  for (size_t i = 0; i < thread_num1; ++i) {
    auto block = [&, i]() {
      task1(i, i + 1);
      return common::SUCCESS;
    };
    tasks1.emplace_back(block);
  }
  common::ThreadPool::GetInstance().SyncRun(tasks1);
  for (size_t i = 0; i < thread_num2; ++i) {
    auto block = [&, i]() {
      task2(i, i + 1);
      return common::SUCCESS;
    };
    tasks2.emplace_back(block);
  }
  common::ThreadPool::GetInstance().SyncRun(tasks2);
}

void LayerNormGradCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 5) {
    MS_LOG(EXCEPTION) << "LayerNormGradCPUKernel needs 5 inputs, but gets " << input_num;
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 3) {
    MS_LOG(EXCEPTION) << "LayerNormGradCPUKernel expects 3 output, but gets" << output_num;
  }
}
}  // namespace kernel
}  // namespace mindspore
