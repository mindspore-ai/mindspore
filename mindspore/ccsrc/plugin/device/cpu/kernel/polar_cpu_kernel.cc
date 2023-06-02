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

#include "plugin/device/cpu/kernel/polar_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <functional>
#include <cmath>
#include <tuple>
#include <type_traits>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
constexpr size_t kPolarInputsNum = 2;
constexpr size_t kPolarOutputsNum = 1;

#define POLAR_COMPUTE_CASE(DTYPE, TYPE)        \
  case (DTYPE): {                              \
    ret = LaunchKernel<TYPE>(inputs, outputs); \
    break;                                     \
  }
}  // namespace

namespace mindspore {
namespace kernel {
bool PolarCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  input1_dtype_ = inputs[0]->GetDtype();
  return true;
}

bool PolarCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                               const std::vector<AddressPtr> &outputs) {
  bool ret = true;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPolarInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPolarOutputsNum, kernel_name_);
  switch (input1_dtype_) {
    POLAR_COMPUTE_CASE(kNumberTypeFloat32, float)
    POLAR_COMPUTE_CASE(kNumberTypeFloat64, double)
    default:
      ret = false;
      MS_EXCEPTION(TypeError) << "For Polar, unsupported input data type: " << TypeIdToString(input1_dtype_) << ".";
  }
  return ret;
}

template <typename T>
bool PolarCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const auto abs = static_cast<T *>(inputs[0]->addr);
  const auto angle = static_cast<T *>(inputs[1]->addr);
  auto output_addr = static_cast<std::complex<T> *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(std::complex<T>);
  auto task = [output_addr, abs, angle](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      output_addr[i].real(abs[i] * cos(angle[i]));
      output_addr[i].imag(abs[i] * sin(angle[i]));
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

std::vector<KernelAttr> PolarCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeComplex64),
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeComplex128)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Polar, PolarCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
