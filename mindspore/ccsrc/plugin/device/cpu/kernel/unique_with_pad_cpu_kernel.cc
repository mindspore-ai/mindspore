/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/unique_with_pad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool UniqueWithPadCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniqueWithPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniqueWithPadOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeInt32) {
    UniqueCpuKernelMod::LaunchKernel<int, int>(inputs, workspace, outputs);
    PadOutput<int>(inputs, outputs, output_sizes_);
  } else if (dtype_ == kNumberTypeInt64) {
    UniqueCpuKernelMod::LaunchKernel<int64_t, int64_t>(inputs, workspace, outputs);
    PadOutput<int64_t>(inputs, outputs, output_sizes_);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16) {
    UniqueCpuKernelMod::LaunchKernel<float, int>(inputs, workspace, outputs);
    PadOutput<float>(inputs, outputs, output_sizes_);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input must be float16, float32, int32, or int64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UniqueWithPad, UniqueWithPadCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
