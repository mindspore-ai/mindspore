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

#include "plugin/device/cpu/kernel/nextafter_cpu_kernel.h"
#include <cmath>
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNextAfterInputsNum = 2;
constexpr size_t kNextAfterOutputsNum = 1;
}  // namespace

bool NextAfterCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNextAfterInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNextAfterOutputsNum, kernel_name_);
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

template <typename T>
bool NextAfterCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                         const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != kNextAfterInputsNum || outputs.size() != kNextAfterOutputsNum) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the operator should have 2 inputs and 1 outputs, but got "
                            << inputs.size() << "input(s) and " << outputs.size() << "output(s)";
  }
  T *x1 = GetDeviceAddress<T>(inputs, kIndex0);
  T *x2 = GetDeviceAddress<T>(inputs, kIndex1);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(x1);
  MS_EXCEPTION_IF_NULL(x2);
  MS_EXCEPTION_IF_NULL(output);

  size_t elem_num = inputs[0]->size / sizeof(T);

  for (size_t i = 0; i < elem_num; i++) {
    output[i] = nextafter(x1[i], x2[i]);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, NextAfterCpuKernelMod::KernelRunFunc>> &NextAfterCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, NextAfterCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &NextAfterCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &NextAfterCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NextAfter, NextAfterCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
