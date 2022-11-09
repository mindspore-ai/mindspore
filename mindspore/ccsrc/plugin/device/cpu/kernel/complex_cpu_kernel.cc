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

#include "plugin/device/cpu/kernel/complex_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <functional>
#include <cmath>
#include <tuple>
#include <type_traits>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
constexpr size_t kComplexInputsNum = 2;
constexpr size_t kComplexOutputsNum = 1;
}  // namespace

namespace mindspore {
namespace kernel {
bool ComplexCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool ComplexCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto input_1 = reinterpret_cast<T *>(inputs[0]->addr);
  const auto input_2 = reinterpret_cast<T *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<std::complex<T> *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(std::complex<T>);
  auto task = [output_addr, input_1, input_2](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      output_addr[i] = std::complex<T>(input_1[i], input_2[i]);
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, ComplexCpuKernelMod::ComplexLaunchFunc>> ComplexCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
   &ComplexCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeComplex128),
   &ComplexCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ComplexCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ComplexCpuKernelMod::ComplexLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Complex, ComplexCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
