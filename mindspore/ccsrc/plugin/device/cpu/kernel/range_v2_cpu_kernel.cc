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

#include "plugin/device/cpu/kernel/range_v2_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kRangeV2InputsNum = 3;
constexpr size_t kRangeV2OutputsNum = 1;
}  // namespace

bool RangeV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRangeV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRangeV2OutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

template <typename T>
bool RangeV2CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                       const std::vector<AddressPtr> &outputs) {
  auto start = static_cast<T *>(inputs[kIndex0]->addr)[kIndex0];
  auto delta = static_cast<T *>(inputs[kIndex2]->addr)[kIndex0];

  auto output_addr = static_cast<T *>(outputs[kIndex0]->addr);
  size_t elem_num = outputs[kIndex0]->size / sizeof(T);
  T val = start;
  for (size_t i = 0; i < elem_num; i++) {
    output_addr[i] = val;
    val += delta;
  }
  return true;
}

const std::vector<std::pair<KernelAttr, RangeV2CpuKernelMod::KernelRunFunc>> &RangeV2CpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, RangeV2CpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &RangeV2CpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &RangeV2CpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &RangeV2CpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &RangeV2CpuKernelMod::LaunchKernel<int64_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RangeV2, RangeV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
