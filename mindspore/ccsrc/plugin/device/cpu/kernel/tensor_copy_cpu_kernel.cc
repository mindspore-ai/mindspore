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

#include "plugin/device/cpu/kernel/tensor_copy_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
void TensorCopyCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);

  auto input_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  auto input_shapes = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  auto output_shapes = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  if ((input_type != output_type) || (input_shapes != output_shapes)) {
    MS_LOG(EXCEPTION) << "The input and output check invalid, kernel: " << kernel_node->fullname_with_scope();
  }

  auto copy_size = GetTypeByte(TypeIdToType(input_type));
  copy_size = std::accumulate(input_shapes.begin(), input_shapes.end(), copy_size, std::multiplies<size_t>());

  input_size_list_.push_back(copy_size);
  output_size_list_.push_back(copy_size);
}

bool TensorCopyCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> & /* workspace */,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  auto input = GetDeviceAddress<void>(inputs, 0);
  auto output = GetDeviceAddress<void>(outputs, 0);

  auto ret = memcpy_s(output, outputs[0]->size, input, inputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Output memcpy error: " << ret;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorMove, TensorCopyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
