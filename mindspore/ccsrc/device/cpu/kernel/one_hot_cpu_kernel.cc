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
#include "device/cpu/kernel/one_hot_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace device {
namespace cpu {
void OneHotCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (output_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "invalid output shape size: " << output_shape.size();
  }
  int axis = AnfAlgo::GetNodeAttr<int>(kernel_node, AXIS);
  if (axis != -1 && IntToSize(axis) >= output_shape.size()) {
    MS_LOG(EXCEPTION) << "invalid axis: " << axis;
  }
  if (axis == -1) {
    axis_ = output_shape.size() - 1;
  } else {
    axis_ = IntToSize(axis);
  }
  depth_ = output_shape[axis_];
  stride_ = 1;
  for (size_t i = axis_ + 1; i < output_shape.size(); ++i) {
    stride_ *= output_shape[i];
  }
}

bool OneHotCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() < 3 || outputs.empty()) {
    MS_LOG(EXCEPTION) << "input or output invalid!";
  }
  auto indices = reinterpret_cast<int *>(inputs[0]->addr);
  auto on_value = reinterpret_cast<float *>(inputs[1]->addr)[0];
  auto off_value = reinterpret_cast<float *>(inputs[2]->addr)[0];
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(int);

  for (size_t i = 0; i < elem_num; i++) {
    size_t stride_num = i / stride_;
    size_t output_index = stride_num * depth_ * stride_ + i % stride_;
    size_t index = IntToSize(indices[i]);
    for (size_t j = 0; j < depth_; j++) {
      if (index == j) {
        output[output_index] = on_value;
      } else {
        output[output_index] = off_value;
      }
      output_index += stride_;
    }
  }

  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
