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
#include "plugin/device/cpu/kernel/complex_abs_cpu_kernel.h"
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool ComplexAbsCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  kernel_name_ = base_operator->GetPrim()->name();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

template <typename T, typename T2>
bool ComplexAbsCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T2 *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T2);
  auto task = [output_addr, input_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      T2 a = input_addr[i].real();
      T2 b = input_addr[i].imag();
      output_addr[i] = sqrt(b * b + a * a);
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, ComplexAbsCpuKernelMod::KernelRunFunc>> &ComplexAbsCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, ComplexAbsCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
     &ComplexAbsCpuKernelMod::LaunchKernel<std::complex<float>, float>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
     &ComplexAbsCpuKernelMod::LaunchKernel<std::complex<double>, double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ComplexAbs, ComplexAbsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
