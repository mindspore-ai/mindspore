/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/transpose_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <unordered_set>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kMaxDim = 10;
}

void TransposeCPUFwdKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  axes_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "perm");
  CheckParameter();
  dtype_ = AnfAlgo ::GetPrevNodeOutputDeviceDataType(kernel_node, 0);
  if (dtype_ == kTypeUnknown) {
    dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  }

  launch_map_[kNumberTypeInt8] = &TransposeCPUFwdKernel::LaunchKernel<int8_t>;
  launch_map_[kNumberTypeInt16] = &TransposeCPUFwdKernel::LaunchKernel<int16_t>;
  launch_map_[kNumberTypeInt32] = &TransposeCPUFwdKernel::LaunchKernel<int>;
  launch_map_[kNumberTypeInt64] = &TransposeCPUFwdKernel::LaunchKernel<int64_t>;
  launch_map_[kNumberTypeUInt8] = &TransposeCPUFwdKernel::LaunchKernel<uint8_t>;
  launch_map_[kNumberTypeUInt16] = &TransposeCPUFwdKernel::LaunchKernel<uint16_t>;
  launch_map_[kNumberTypeUInt32] = &TransposeCPUFwdKernel::LaunchKernel<uint32_t>;
  launch_map_[kNumberTypeUInt64] = &TransposeCPUFwdKernel::LaunchKernel<uint64_t>;
  launch_map_[kNumberTypeFloat32] = &TransposeCPUFwdKernel::LaunchKernel<float>;
  launch_map_[kNumberTypeBool] = &TransposeCPUFwdKernel::LaunchKernel<bool>;

  auto iter = launch_map_.find(dtype_);
  if (iter != launch_map_.end()) {
    launch_func_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Input data type: " << dtype_ << "is not supported for Transpose kernel on CPU.";
  }
}

bool TransposeCPUFwdKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> & /*workspace*/,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  launch_func_(this, inputs, outputs);
  return true;
}

void TransposeCPUFwdKernel::CheckParameter() const {
  if (input_shape_.size() > kMaxDim) {
    MS_LOG(EXCEPTION) << "Input tensor is " << input_shape_.size() << ", out of bound max dimension 10";
  }

  if (input_shape_.empty()) {
    MS_LOG(EXCEPTION) << "Input tensor is empty";
  }

  if (input_shape_.size() != axes_.size()) {
    MS_LOG(EXCEPTION) << "Input perm size is not equal with input shape";
  }

  // Input axes include the same axis
  std::unordered_set<int64_t> unique_axes{axes_.begin(), axes_.end()};
  if (unique_axes.size() != axes_.size()) {
    MS_LOG(EXCEPTION) << "Input perm is illegal, it has the same axis";
  }

  // Input axes not in ture range(input_shape_.size())
  int64_t shape_size = input_shape_.size();
  for (auto &axis : axes_) {
    if (axis < 0 || axis >= shape_size) {
      MS_LOG(EXCEPTION) << "Input perm axis is out of bound input shape size";
    }
  }
}

template <typename T>
void TransposeCPUFwdKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  int dimension = input_shape_.size();
  // Calculate input tensor strides
  std::array<uint32_t, kMaxDim> input_strides{0};
  input_strides[dimension - 1] = 1;
  for (int i = dimension - 2; i >= 0; --i) {
    input_strides[i] = input_shape_[i + 1] * input_strides[i + 1];
  }

  // Calculate output strides and back strides
  std::array<uint32_t, kMaxDim> strides{0};
  std::array<uint32_t, kMaxDim> back_strides{0};
  for (int i = dimension - 1; i >= 0; --i) {
    strides[i] = input_strides[axes_[i]];
    back_strides[i] = (output_shape_[i] - 1) * strides[i];
  }

  std::array<uint32_t, kMaxDim> coordinates{0};
  auto get_next_pos = [&coordinates, &strides, &back_strides, &dimension, this](int curr_pos) {
    for (int i = dimension - 1; i >= 0; --i) {
      if (coordinates[i] + 1 == output_shape_[i]) {
        coordinates[i] = 0;
        curr_pos -= back_strides[i];
      } else {
        coordinates[i]++;
        curr_pos += strides[i];
        break;
      }
    }
    return curr_pos;
  };

  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t size = IntToSize(inputs[0]->size / sizeof(T));
  output[0] = input[0];
  int pos = 0;
  for (size_t i = 1; i < size; ++i) {
    pos = get_next_pos(pos);
    output[i] = input[pos];
  }
}
}  // namespace kernel
}  // namespace mindspore
