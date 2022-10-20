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

#include <cmath>
#include <string>
#include <algorithm>
#include "plugin/device/cpu/kernel/blackman_window_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/blackman_window.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBlackmanWindowInputsNum = 1;
constexpr size_t kBlackmanWindowOutputsNum = 1;
}  // namespace

bool BlackmanWindowCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBlackmanWindowInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBlackmanWindowOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BlackmanWindow>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  periodic_ = kernel_ptr->get_periodic();
  return true;
}

template <typename T1, typename T2>
bool BlackmanWindowCpuKernelMod::BlackmanWindowKernelFunc(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &,
                                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto input = static_cast<T1 *>(inputs[0]->addr);
  auto output = static_cast<T2 *>(outputs[0]->addr);

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
    const double PI = 3.14159265358979323846;
    const double x = static_cast<double>(window_length);
    for (size_t i = 0; i < pre_window_length; i++) {
      auto temp = static_cast<T2>(0.08 * cos((4 * PI * i) / (x - 1)) - 0.5 * cos((2 * PI * i) / (x - 1)) + 0.42);
      *(output + i) = temp;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, BlackmanWindowCpuKernelMod::BlackmanWindowFunc>>
  BlackmanWindowCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &BlackmanWindowCpuKernelMod::BlackmanWindowKernelFunc<int32_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &BlackmanWindowCpuKernelMod::BlackmanWindowKernelFunc<int32_t, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &BlackmanWindowCpuKernelMod::BlackmanWindowKernelFunc<int32_t, double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &BlackmanWindowCpuKernelMod::BlackmanWindowKernelFunc<int64_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &BlackmanWindowCpuKernelMod::BlackmanWindowKernelFunc<int64_t, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &BlackmanWindowCpuKernelMod::BlackmanWindowKernelFunc<int64_t, double>}};

std::vector<KernelAttr> BlackmanWindowCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BlackmanWindowFunc> &pair) { return pair.first; });

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BlackmanWindow, BlackmanWindowCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
