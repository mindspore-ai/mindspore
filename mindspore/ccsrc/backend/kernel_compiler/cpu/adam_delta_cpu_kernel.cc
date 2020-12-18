/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/adam_delta_cpu_kernel.h"
#include <thread>
#include <vector>
#include <string>
#include <memory>
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
constexpr size_t kAdamDeltaInputSize = 9;
namespace {
struct ComputeParam {
  float *delta_{nullptr};
  float *m_{nullptr};
  float *v_{nullptr};
  float *grad_{nullptr};
  float beta1_{0};
  float beta2_{0};
  float epsilon_{0};
  float lr_{0};
  bool use_nesterov_{0};
};

void ComputeWeightDelta(const std::shared_ptr<ComputeParam> &input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  MS_EXCEPTION_IF_NULL(input_params->delta_);
  MS_EXCEPTION_IF_NULL(input_params->m_);
  MS_EXCEPTION_IF_NULL(input_params->v_);
  MS_EXCEPTION_IF_NULL(input_params->grad_);
  auto delta = input_params->delta_;
  auto m = input_params->m_;
  auto v = input_params->v_;
  auto lr = input_params->lr_;
  auto beta1 = input_params->beta1_;
  auto beta2 = input_params->beta2_;
  auto epsilon = input_params->epsilon_;
  auto use_nesterov = input_params->use_nesterov_;
  auto grad = input_params->grad_;
  for (size_t i = start; i < end; ++i) {
    m[i] *= beta1;
    v[i] *= beta2;
    m[i] += (1 - beta1) * grad[i];
    v[i] += (1 - beta2) * grad[i] * grad[i];
    if (use_nesterov) {
      delta[i] = -lr * (m[i] * beta1 + (1 - beta1) * grad[i]) / (std::sqrt(v[i]) + epsilon);
    } else {
      delta[i] = -lr * m[i] / (std::sqrt(v[i]) + epsilon);
    }
  }
}
}  // namespace

void AdamDeltaCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> delta_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  std::vector<size_t> m_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> v_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> grad_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 8);
  if (!IsSameShape(delta_shape, m_shape)) {
    MS_LOG(EXCEPTION) << "Delta and m should have the same shape";
  }
  if (!IsSameShape(delta_shape, v_shape)) {
    MS_LOG(EXCEPTION) << "Delta and v should have the same shape";
  }
  if (!IsSameShape(delta_shape, grad_shape)) {
    MS_LOG(EXCEPTION) << "Delta and grad should have the same shape";
  }
  if (delta_shape.empty()) {
    MS_LOG(EXCEPTION) << "Delta must be at least 1D";
  }
  elem_num_ = 1;
  for (size_t i = 0; i < delta_shape.size(); ++i) {
    elem_num_ *= delta_shape[i];
  }
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "Invalid delta shape";
  }
  if (AnfAlgo::HasNodeAttr(USE_NESTEROV, kernel_node)) {
    use_nesterov_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "use_nesterov");
  }
}

void AdamDeltaCPUKernel::CheckParams(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &workspace,
                                     const std::vector<kernel::AddressPtr> &outputs) const {
  if (inputs.size() != kAdamDeltaInputSize) {
    MS_LOG(EXCEPTION) << "Error input size!";
  }
  size_t elem_size = elem_num_ * 4;
  std::vector<size_t> expect_sizes = {elem_size, elem_size, 4, 4, 4, 4, 4, 4, elem_size};
  std::vector<std::string> input_names = {"m",     "v",     "beta1_power", "beta2_power", "lr",
                                          "beta1", "beta2", "epsilon",     "grad"};
  for (size_t i = 0; i < kAdamDeltaInputSize; ++i) {
    if (inputs[i]->size != expect_sizes[i]) {
      MS_LOG(EXCEPTION) << "Error input " << input_names[i] << " size!";
    }
  }
  if (outputs.size() < 1 || outputs[0]->size != elem_size) {
    MS_LOG(EXCEPTION) << "Error output delta size!";
  }
}

bool AdamDeltaCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CheckParams(inputs, workspace, outputs);
  auto m = reinterpret_cast<float *>(inputs[0]->addr);
  auto v = reinterpret_cast<float *>(inputs[1]->addr);
  auto beta1_power = reinterpret_cast<float *>(inputs[2]->addr)[0];
  if (beta1_power == 1) {
    MS_LOG(EXCEPTION) << "The beta1_power should not be 1";
  }
  auto beta2_power = reinterpret_cast<float *>(inputs[3]->addr)[0];
  auto lr = reinterpret_cast<float *>(inputs[4]->addr)[0];
  auto beta1 = reinterpret_cast<float *>(inputs[5]->addr)[0];
  auto beta2 = reinterpret_cast<float *>(inputs[6]->addr)[0];
  auto epsilon = reinterpret_cast<float *>(inputs[7]->addr)[0];
  auto grad = reinterpret_cast<float *>(inputs[8]->addr);
  auto delta = reinterpret_cast<float *>(outputs[0]->addr);
  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);
  size_t thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  if (elem_num_ < thread_num) {
    thread_num = elem_num_;
  }
  std::vector<common::Task> tasks;
  std::vector<std::shared_ptr<ComputeParam>> thread_params;
  tasks.reserve(thread_num);

  size_t end = 0;
  size_t offset = elem_num_ / thread_num;
  size_t left = elem_num_ % thread_num;
  for (size_t i = 0; i < thread_num; ++i) {
    auto params = std::make_shared<ComputeParam>();
    params->delta_ = delta;
    params->m_ = m;
    params->v_ = v;
    params->grad_ = grad;
    params->beta1_ = beta1;
    params->beta2_ = beta2;
    params->use_nesterov_ = use_nesterov_;
    params->lr_ = lr;
    params->epsilon_ = epsilon;
    size_t start = end;
    end = start + offset;
    if (i < left) {
      end += 1;
    }
    auto task = [&params, start, end]() {
      ComputeWeightDelta(params, start, end);
      return common::SUCCESS;
    };
    tasks.emplace_back(task);
    thread_params.emplace_back(params);
  }
  common::ThreadPool::GetInstance().SyncRun(tasks);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
