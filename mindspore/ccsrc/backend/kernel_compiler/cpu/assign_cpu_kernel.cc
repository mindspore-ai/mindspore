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
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void AssignCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  auto input_x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto input_y_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    batch_size_ *= input_x_shape[i];
  }

  if (input_x_shape.size() != input_y_shape.size()) MS_LOG(EXCEPTION) << "x y must be same shape";
  for (size_t i = 0; i < input_x_shape.size(); ++i) {
    if (input_x_shape[i] != input_y_shape[i]) {
      MS_LOG(EXCEPTION) << "x y must be same shape";
    }
  }
  input_x_dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  if (input_x_dtype_ == kNumberTypeFloat32 || input_x_dtype_ == kNumberTypeInt32) {
    input_x_dtype_size_ = 4;
  } else if (input_x_dtype_ == kNumberTypeFloat64 || input_x_dtype_ == kNumberTypeInt64) {
    input_x_dtype_size_ = 8;
  } else {
    MS_LOG(EXCEPTION) << "input_x dtype only support float32, float64, int32, int64";
  }
}

bool AssignCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "indices dtype only support float32, float64, int32, int64";
    return false;
  }
  return true;
}

template <typename T>
void AssignCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  T *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_y = reinterpret_cast<T *>(inputs[1]->addr);
  auto max_size = inputs[0]->size;
  size_t total_size = input_x_dtype_size_ * batch_size_;
  if (total_size > max_size) {
    MS_LOG(EXCEPTION) << "Memcpy size must <= max_size, but got memcpy size is : " << total_size
                      << ", max size is : " << max_size;
  }
  int ret = memcpy_s(input_x, total_size, input_y, total_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
  }
}
}  // namespace kernel
}  // namespace mindspore
