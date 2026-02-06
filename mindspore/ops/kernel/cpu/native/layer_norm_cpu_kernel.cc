/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/layer_norm_cpu_kernel.h"
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

#include "include/utils/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace layer_norm_cpu {
namespace {
constexpr size_t kLayerNormInputsNum = 6;
constexpr size_t kLayerNormOutputsNum = 3;
constexpr size_t kLayerNormInputXIndex = 0;
constexpr size_t kLayerNormInputGammaIndex = 1;
constexpr size_t kLayerNormInputBetaIndex = 2;
constexpr size_t kLayerNormInputBeginNormAxisIndex = 3;
constexpr size_t kLayerNormInputBeginParamsAxisIndex = 4;
constexpr size_t kLayerNormInputEpsilonIndex = 5;
constexpr size_t kLayerNormOutputYIndex = 0;
constexpr size_t kLayerNormOutputMeanIndex = 1;
constexpr size_t kLayerNormOutputVarIndex = 2;
}  // namespace
bool LayerNormCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  eps_ = inputs[kLayerNormInputEpsilonIndex]->GetValueWithCheck<float_t>();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int LayerNormCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid LayerNormCpuKernelMod input size!";
  }
  auto x_shape = inputs[kLayerNormInputXIndex]->GetShapeVector();
  auto begin_norm_axis = inputs[kLayerNormInputBeginNormAxisIndex]->GetValueWithCheck<int64_t>();
  auto begin_params_axis = inputs[kLayerNormInputBeginParamsAxisIndex]->GetValueWithCheck<int64_t>();

  if (begin_norm_axis < 0) {
    begin_norm_axis += SizeToLong(x_shape.size());
  }
  if (begin_params_axis < 0) {
    begin_params_axis += SizeToLong(x_shape.size());
  }
  block_num_ = 1;
  block_size_ = 1;
  param_num_ = 1;
  for (size_t i = 0; i < LongToSize(begin_norm_axis); i++) {
    block_num_ *= LongToUlong(x_shape[i]);
  }
  for (size_t i = LongToSize(begin_norm_axis); i < x_shape.size(); i++) {
    block_size_ *= LongToUlong(x_shape[i]);
  }
  for (size_t i = LongToSize(begin_params_axis); i < x_shape.size(); i++) {
    param_num_ *= LongToUlong(x_shape[i]);
  }
  if (block_num_ == 0 || block_size_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'input_x' must be at least 1, but got "
                      << x_shape;
  }
  return ret;
}

bool LayerNormCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLayerNormInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLayerNormOutputsNum, kernel_name_);
  kernel_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void LayerNormCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  size_t f_size = sizeof(T);
  if (inputs[kLayerNormInputGammaIndex]->size() != f_size * param_num_ ||
      inputs[kLayerNormInputBetaIndex]->size() != f_size * param_num_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the product of gamma and beta's shape must be " << param_num_;
  }
  if (outputs[kLayerNormOutputMeanIndex]->size() != outputs[kLayerNormOutputVarIndex]->size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the product of mean and var's shape must be " << block_num_;
  }
  auto x = reinterpret_cast<T *>(inputs[kLayerNormInputXIndex]->device_ptr());
  auto gamma = reinterpret_cast<T *>(inputs[kLayerNormInputGammaIndex]->device_ptr());
  auto beta = reinterpret_cast<T *>(inputs[kLayerNormInputBetaIndex]->device_ptr());
  auto y = reinterpret_cast<T *>(outputs[kLayerNormOutputYIndex]->device_ptr());
  auto mean = reinterpret_cast<float *>(outputs[kLayerNormOutputMeanIndex]->device_ptr());
  auto var = reinterpret_cast<float *>(outputs[kLayerNormOutputVarIndex]->device_ptr());
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(gamma);
  MS_EXCEPTION_IF_NULL(beta);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(var);
  size_t thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  if (block_num_ < thread_num) {
    thread_num = block_num_;
  }
  std::vector<common::Task> tasks;
  tasks.reserve(thread_num);
  auto task = [this, &x, &gamma, &beta, &y, &mean, &var, thread_num](size_t start) {
    for (size_t c = 0; c < ceil(static_cast<double>(block_num_) / thread_num); ++c) {
      if (c * thread_num + start >= block_num_) {
        continue;
      }
      size_t i = c * thread_num + start;
      double sum = 0.0;
      double square_sum = 0.0;
      for (size_t j = i * block_size_; j < (i + 1) * block_size_; ++j) {
        auto x_j = static_cast<double>(x[j]);
        sum += x_j;
        square_sum += x_j * x_j;
      }
      double block_mean = sum / block_size_;
      float block_var = static_cast<float>(square_sum / block_size_ - block_mean * block_mean);
      if (block_var < 0) {
        block_var = 0;
      }
      MS_EXCEPTION_IF_ZERO("Var + Epsilon", block_var + eps_);
      for (size_t j = i * block_size_; j < (i + 1) * block_size_; ++j) {
        auto param_shift = j % param_num_;
        y[j] = (x[j] - static_cast<T>(block_mean)) / static_cast<T>(std::sqrt(block_var + eps_)) * gamma[param_shift] +
               beta[param_shift];
      }
      mean[i] = static_cast<float>(block_mean);
      var[i] = block_var;
    }
  };
  for (size_t i = 0; i < thread_num; ++i) {
    auto block = [&, i]() {
      task(i);
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(block);
  }
  ParallelLaunch(tasks);
}

std::vector<std::pair<KernelAttr, LayerNormCpuKernelMod::KernelFunc>> LayerNormCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &LayerNormCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> LayerNormCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LayerNorm, LayerNormCpuKernelMod);
}  // namespace layer_norm_cpu
}  // namespace kernel
}  // namespace mindspore
