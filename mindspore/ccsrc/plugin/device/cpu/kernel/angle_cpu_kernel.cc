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
#include "plugin/device/cpu/kernel/angle_cpu_kernel.h"
#include <complex>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace {
const size_t kOutputsNum = 1;
const size_t kInputsNum = 1;
}  // namespace

namespace mindspore {
namespace kernel {
bool AngleCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  input_dtype_ = inputs[0]->GetDtype();
  return true;
}

bool AngleCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  bool ret = true;
  switch (input_dtype_) {
    case (kNumberTypeComplex64): {
      ret = LaunchKernel<std::complex<float>, float>(inputs, outputs);
      break;
    }
    case (kNumberTypeComplex128): {
      ret = LaunchKernel<std::complex<double>, double>(inputs, outputs);
      break;
    }
    default: {
      ret = false;
      MS_EXCEPTION(TypeError) << "For 'Angle', unsupported input data type: " << TypeIdToString(input_dtype_);
    }
  }
  return ret;
}

template <typename T, typename T2>
bool AngleCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  auto input_addr = static_cast<T *>(inputs[0]->addr);
  auto output_addr = static_cast<T2 *>(outputs[0]->addr);
  size_t output_size = outputs[0]->size / sizeof(T2);
  auto task = [output_addr, input_addr](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      T2 a = input_addr[i].real();
      T2 b = input_addr[i].imag();
      output_addr[i] = atan2(b, a);
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

std::vector<KernelAttr> AngleCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),

    KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64)};

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Angle, AngleCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
