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

#include "kernel/cpu/transpose_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"
namespace mindspore {
namespace kernel {
const size_t kMaxDim = 100;
void TransposeCPUFwdKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  axis_ = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, "perm");
  if (shape_.size() != axis_.size()) {
    MS_LOG(EXCEPTION) << "The size of input shape and transpose axis shape must be equal.";
  }
}
bool TransposeCPUFwdKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> & /*workspace*/,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  auto input = reinterpret_cast<float *>(inputs[0]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  size_t size = IntToSize(inputs[0]->size / sizeof(float));
  size_t shape_size = IntToSize(shape_.size());
  if (shape_size > kMaxDim) {
    MS_LOG(EXCEPTION) << "Input is " << shape_size << "-D, but transpose supports max " << kMaxDim << "-D inputs.";
  }
  size_t pos_array[kMaxDim];
  size_t size_offset[kMaxDim];
  size_offset[0] = size / shape_[0];
  for (size_t i = 1; i < shape_size; i++) {
    size_offset[i] = size_offset[SizeToInt(i) - 1] / shape_[i];
  }
  for (size_t position = 0; position < size; position += 1) {
    size_t temp_position = position;
    pos_array[0] = temp_position / size_offset[0];
    for (size_t i = 1; i < shape_size; i++) {
      temp_position -= pos_array[SizeToInt(i) - 1] * size_offset[i - 1];
      pos_array[i] = temp_position / size_offset[i];
    }
    size_t new_position = pos_array[axis_[SizeToInt(shape_size) - 1]];
    size_t new_position_size = 1;
    for (int j = shape_size - 2; j >= 0; j--) {
      new_position_size *= shape_[axis_[j + 1]];
      new_position += pos_array[axis_[j]] * new_position_size;
    }
    output[new_position] = input[position];
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
