/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/unique_with_pad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUniqueWithPadInputsNum = 2;
constexpr size_t kUniqueWithPadOutputsNum = 2;
}  // namespace

bool UniqueWithPadCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUniqueWithPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUniqueWithPadOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeInt32) {
    UniqueCPUKernel::LaunchKernel<int, int>(inputs, workspace, outputs);
    PadOutput<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    UniqueCPUKernel::LaunchKernel<int64_t, int64_t>(inputs, workspace, outputs);
    PadOutput<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat16) {
    UniqueCPUKernel::LaunchKernel<float, int>(inputs, workspace, outputs);
    PadOutput<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
  return true;
}

template <typename T>
void UniqueWithPadCPUKernel::PadOutput(const std::vector<AddressPtr> &inputs,
                                       const std::vector<AddressPtr> &outputs) const {
  auto pad_num = *reinterpret_cast<T *>(inputs[1]->addr);
  auto *out = reinterpret_cast<T *>(outputs[0]->addr);
  for (size_t i = output_size_; i < input_size_; ++i) {
    out[i] = pad_num;
  }
}
}  // namespace kernel
}  // namespace mindspore
