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

#include "backend/kernel_compiler/cpu/assign_cpu_kernel.h"
#include <string>
#include <map>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
static std::map<TypeId, size_t> input_x_dtype_size_map = {
  {kNumberTypeBool, sizeof(bool)}, {kNumberTypeInt8, 1},    {kNumberTypeInt16, 2},   {kNumberTypeInt32, 4},
  {kNumberTypeInt64, 8},           {kNumberTypeUInt8, 1},   {kNumberTypeUInt16, 2},  {kNumberTypeUInt32, 4},
  {kNumberTypeUInt64, 8},          {kNumberTypeFloat16, 2}, {kNumberTypeFloat32, 4}, {kNumberTypeFloat64, 8}};

void AssignCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto input_x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto input_y_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (input_x_shape.size() != input_y_shape.size()) MS_LOG(EXCEPTION) << "x y must be same shape";
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    if (input_x_shape[i] != input_y_shape[i]) {
      MS_LOG(EXCEPTION) << "x y must be same shape";
    }
    batch_size_ *= input_x_shape[i];
  }
  input_x_dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  if (input_x_dtype_size_map.find(input_x_dtype_) == input_x_dtype_size_map.end()) {
    MS_LOG(EXCEPTION) << "unsupported input_x dtype";
  }
  input_x_dtype_size_ = input_x_dtype_size_map[input_x_dtype_];
}

bool AssignCPUKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> & /*workspace*/,
                             const std::vector<AddressPtr> &outputs) {
  auto max_size = inputs[0]->size;
  size_t total_size = input_x_dtype_size_ * batch_size_;
  if (total_size > max_size) {
    MS_LOG(EXCEPTION) << "Memcpy size must <= max_size, but got memcpy size is : " << total_size
                      << ", max size is : " << max_size;
  }
  int ret = memcpy_s(inputs[0]->addr, max_size, inputs[1]->addr, total_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, error no " << ret;
  }
  ret = memcpy_s(outputs[0]->addr, max_size, inputs[1]->addr, total_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, error no " << ret;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
