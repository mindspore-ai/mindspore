/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/argmax_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void ArgmaxCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (shape.size() != 2) {
    MS_LOG(EXCEPTION) << "argmax kernel dims invalid " << shape.size();
  }
  batch_size_ = shape[0];
  class_num_ = shape[1];

  int axis = AnfAlgo::GetNodeAttr<int>(kernel_node, AXIS);
  if (axis != -1 && axis != 1) {
    MS_LOG(EXCEPTION) << "argmax kernel not support axis " << axis;
  }
}

bool ArgmaxCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspaces*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "input or output empty!";
  }

  size_t batch_float_size = batch_size_ * sizeof(float);
  size_t batch_class_float_size = class_num_ * batch_float_size;
  if (inputs[0]->size != batch_class_float_size || outputs[0]->size != batch_float_size) {
    MS_LOG(EXCEPTION) << "invalid input or output data size!";
  }
  auto input = reinterpret_cast<float *>(inputs[0]->addr);
  auto output = reinterpret_cast<int *>(outputs[0]->addr);
  size_t row_start = 0;
  for (size_t i = 0; i < batch_size_; ++i) {
    size_t max_index = 0;
    float max_value = input[row_start];
    for (size_t j = 1; j < class_num_; ++j) {
      size_t index = row_start + j;
      if (input[index] > max_value) {
        max_value = input[index];
        max_index = j;
      }
    }
    output[i] = SizeToInt(max_index);
    row_start += class_num_;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
