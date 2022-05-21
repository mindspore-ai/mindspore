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

void RangeV2CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
}

bool RangeV2CpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRangeV2InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRangeV2OutputsNum, kernel_name_);
  switch (dtype_) {
    case kNumberTypeInt32:
      LaunchKernel<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      LaunchKernel<int64_t>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      LaunchKernel<float>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      LaunchKernel<double>(inputs, outputs);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                              << "', the dtype of input must be int32, int64, float32, float64, but got "
                              << TypeIdLabel(dtype_) << ".";
      break;
  }
  return true;
}

template <typename T>
void RangeV2CpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                       const std::vector<AddressPtr> &outputs) const {
  auto start = reinterpret_cast<T *>(inputs[kIndex0]->addr)[kIndex0];
  auto delta = reinterpret_cast<T *>(inputs[kIndex2]->addr)[kIndex0];

  auto output_addr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  size_t elem_num = outputs[kIndex0]->size / sizeof(T);
  T val = start;
  for (size_t i = 0; i < elem_num; i++) {
    output_addr[i] = val;
    val += delta;
  }
}

std::vector<KernelAttr> RangeV2CpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RangeV2, RangeV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
