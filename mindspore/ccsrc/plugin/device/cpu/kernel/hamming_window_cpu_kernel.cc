/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/hamming_window_cpu_kernel.h"
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <functional>
#include "mindspore/core/ops/hamming_window.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/arithmetic_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kHammingWindowOutputNum = 1;
const size_t kHammingWindowInputNum = 1;
}  // namespace

bool HammingWindowCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHammingWindowInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHammingWindowOutputNum, kernel_name_);
  auto op_prim = std::dynamic_pointer_cast<ops::HammingWindow>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  periodic_ = op_prim->get_periodic();
  alpha_ = op_prim->get_alpha();
  beta_ = op_prim->get_beta();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "HammingWindow does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T, typename S>
bool HammingWindowCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> & /* workspace */,
                                             const std::vector<AddressPtr> &outputs) const {
  auto *length_addr = static_cast<T *>(inputs[0]->addr);
  auto *output = static_cast<S *>(outputs[0]->addr);
  int64_t window_length_ = static_cast<int64_t>(*length_addr);
  if (window_length_ < 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the value of input 'length' cannot be negative, but got "
                             << window_length_;
  } else if (window_length_ == 0) {
    return true;
  } else if (window_length_ == 1) {
    *output = S{1};
    return true;
  }
  int64_t length = periodic_ ? window_length_ : (window_length_ - 1);
  constexpr double t_pi = 6.283185307179586476925286766559;
  auto func = [length, alpha = alpha_, beta = beta_, t_pi, &output](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      double result = alpha - beta * std::cos(i * t_pi / length);
      output[i] = static_cast<S>(result);
    }
  };
  ParallelLaunch(func, LongToSize(window_length_));
  return true;
}

std::vector<std::pair<KernelAttr, HammingWindowCpuKernelMod::HammingWindowFunc>> HammingWindowCpuKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<int8_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<int16_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<int32_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<int64_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<uint8_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<uint16_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<uint32_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat16),
    &HammingWindowCpuKernelMod::LaunchKernel<uint64_t, float16>},
   {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<int8_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<int16_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<int32_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<int64_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<uint8_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<uint16_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<uint32_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32),
    &HammingWindowCpuKernelMod::LaunchKernel<uint64_t, float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<int8_t, double>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<int16_t, double>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<int32_t, double>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<int64_t, double>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<uint8_t, double>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<uint16_t, double>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<uint32_t, double>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat64),
    &HammingWindowCpuKernelMod::LaunchKernel<uint64_t, double>}};

std::vector<KernelAttr> HammingWindowCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, HammingWindowFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HammingWindow, HammingWindowCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
