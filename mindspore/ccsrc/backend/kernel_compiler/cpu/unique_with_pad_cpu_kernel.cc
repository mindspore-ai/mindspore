/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
bool UniqueWithPadCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    UniqueCPUKernel::LaunchKernel<int, int>(inputs, workspace, outputs);
    PadOutput<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    UniqueCPUKernel::LaunchKernel<int64_t, int>(inputs, workspace, outputs);
    PadOutput<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    UniqueCPUKernel::LaunchKernel<float, int>(inputs, workspace, outputs);
    PadOutput<float>(inputs, outputs);
  }
  return true;
}

template <typename T>
void UniqueWithPadCPUKernel::PadOutput(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (inputs.size() < 2) {
    MS_LOG(EXCEPTION) << "Input size should be large than 1";
  }
  if (outputs.size() < 1) {
    MS_LOG(EXCEPTION) << "Output size should be large than 0";
  }
  T pad_num = *reinterpret_cast<T *>(inputs[1]->addr);
  T *out = reinterpret_cast<T *>(outputs[0]->addr);
  for (size_t i = output_size_; i < input_size_; ++i) {
    out[i] = pad_num;
  }
}

void UniqueWithPadCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but UniqueCPUKernel only support 1d.";
  }
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but UniqueCPUKernel needs 2 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but UniqueCPUKernel needs 2 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
