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

#include <cmath>
#include <string>
#include <algorithm>
#include "plugin/device/cpu/kernel/bartlett_window_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bartlett_window.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBartlettWindowInputsNum = 1;
constexpr size_t kBartlettWindowOutputsNum = 1;
}  // namespace

bool BartlettWindowCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBartlettWindowInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBartlettWindowOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BartlettWindow>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  periodic_ = kernel_ptr->get_periodic();
  return true;
}

template <typename T1, typename T2>
bool BartlettWindowCpuKernelMod::BartlettWindowKernelFunc(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &,
                                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto input = reinterpret_cast<T1 *>(inputs[0]->addr);
  auto output = reinterpret_cast<T2 *>(outputs[0]->addr);

  if (*input < 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', input window_length should be >= 0, but got " << *input;
  }

  auto window_length = static_cast<int64_t>(*input);
  double pre_window_length = static_cast<double>(window_length);
  const size_t OUTPUTISONE = 1;

  if (*input == 1) {
    *output = static_cast<T2>(OUTPUTISONE);
  } else {
    if (periodic_) {
      window_length += 1;
    }
    const size_t first_half_size = static_cast<size_t>((window_length - 1) / 2);
    const double x = static_cast<double>(window_length);
    for (size_t i = 0; i <= first_half_size; i++) {
      auto value = static_cast<T2>((2. * i) / (x - 1.));
      *(output + i) = value;
    }
    for (size_t i = first_half_size + 1; i < pre_window_length; i++) {
      auto value = static_cast<T2>(2. - (2. * i) / (x - 1.));
      *(output + i) = value;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, BartlettWindowCpuKernelMod::BartlettWindowFunc>>
  BartlettWindowCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &BartlettWindowCpuKernelMod::BartlettWindowKernelFunc<int32_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &BartlettWindowCpuKernelMod::BartlettWindowKernelFunc<int32_t, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &BartlettWindowCpuKernelMod::BartlettWindowKernelFunc<int32_t, double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &BartlettWindowCpuKernelMod::BartlettWindowKernelFunc<int64_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &BartlettWindowCpuKernelMod::BartlettWindowKernelFunc<int64_t, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &BartlettWindowCpuKernelMod::BartlettWindowKernelFunc<int64_t, double>}};

std::vector<KernelAttr> BartlettWindowCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BartlettWindowFunc> &pair) { return pair.first; });

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BartlettWindow, BartlettWindowCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
