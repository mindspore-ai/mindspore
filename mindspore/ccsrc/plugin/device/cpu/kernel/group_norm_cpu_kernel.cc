/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/group_norm_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGroupNormInputsNum = 5;
constexpr size_t kGroupNormOutputsNum = 3;
constexpr size_t kNumberTwo = 2;
}  // namespace
bool GroupNormCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GroupNormCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid GroupNormCpuKernelMod input size!";
  }

  auto x_shape = inputs[kIndex0]->GetShapeVector();
  if (x_shape.size() < kNumberTwo) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dims of input tesnor must be not less than 2 "
                      << "but got: " << x_shape.size();
  }

  auto batch = x_shape[0];
  auto num_groups = inputs[kIndex1]->GetValueWithCheck<int64_t>();

  num_channel_ = x_shape[1];
  HxW_ = LongToSize((x_shape.size() == kNumberTwo)
                      ? 1
                      : std::accumulate(x_shape.begin() + kIndex2, x_shape.end(), 1, std::multiplies<int64_t>()));
  eps_ = inputs[kIndex4]->GetValueWithCheck<float_t>();
  inner_size_ = LongToSize(num_channel_ * HxW_ / num_groups);
  outter_size_ = LongToSize(batch * num_groups);

  if (num_channel_ % num_groups != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'num_channels' must be divided by 'num_groups', "
                      << "but got 'num_channels': " << num_channel_ << " ,'num_groups': " << num_groups;
  }

  return ret;
}

bool GroupNormCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGroupNormInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGroupNormOutputsNum, kernel_name_);
  kernel_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void GroupNormCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  size_t f_size = sizeof(T);
  if (inputs[kIndex2]->size() != f_size * LongToUlong(num_channel_) ||
      inputs[kIndex3]->size() != f_size * LongToUlong(num_channel_)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the product of gamma and beta's shape must be " << num_channel_;
  }
  auto x = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto gamma = reinterpret_cast<T *>(inputs[kIndex2]->device_ptr());
  auto beta = reinterpret_cast<T *>(inputs[kIndex3]->device_ptr());
  auto y = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  auto mean = reinterpret_cast<T *>(outputs[kIndex1]->device_ptr());
  auto rstd = reinterpret_cast<T *>(outputs[kIndex2]->device_ptr());
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(gamma);
  MS_EXCEPTION_IF_NULL(beta);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(rstd);

  auto task = [this, &x, &gamma, &beta, &y, &mean, &rstd](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      double sum = 0.0;
      double sum_square = 0.0;
      for (size_t j = i * inner_size_; j < (i + 1) * inner_size_; ++j) {
        sum += static_cast<double>(x[j]);
        sum_square += static_cast<double>(x[j]) * static_cast<double>(x[j]);
      }
      double mean_val = sum / inner_size_;
      double rstd_val = std::sqrt(1 / ((sum_square / inner_size_ - mean_val * mean_val) + static_cast<double>(eps_)));
      for (size_t j = i * inner_size_; j < (i + 1) * inner_size_; ++j) {
        auto param_index = (j / HxW_) % num_channel_;
        y[j] = (x[j] - static_cast<T>(mean_val)) * static_cast<T>(rstd_val) * gamma[param_index] + beta[param_index];
      }
      mean[i] = static_cast<T>(mean_val);
      rstd[i] = static_cast<T>(rstd_val);
    }
  };
  ParallelLaunchAutoSearch(task, outter_size_, this, &parallel_search_info_);
}

std::vector<std::pair<KernelAttr, GroupNormCpuKernelMod::KernelFunc>> GroupNormCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &GroupNormCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GroupNormCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GroupNormCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> GroupNormCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GroupNorm, GroupNormCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
