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
#include <map>
#include <string>
#include <vector>
#include "kernel/cpu/reduce_cpu_kernel.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
const size_t kReduceTypeMax = 0;
const size_t kReduceTypeMean = 1;
const size_t kReduceTypeSum = 2;
const size_t kMaxDim = 100;
void ReduceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name == "ReduceMax") {
    reduce_type_ = kReduceTypeMax;
  } else if (kernel_name == "ReduceMean") {
    reduce_type_ = kReduceTypeMean;
  } else if (kernel_name == "ReduceSum") {
    reduce_type_ = kReduceTypeSum;
  } else {
    MS_LOG(EXCEPTION) << "Array reduce kernel type " << kernel_name << " is not supported.";
  }
  shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto axis_addr = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(AXIS);
  if (axis_addr->isa<ValueTuple>()) {
    auto attr_axis = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, AXIS);
    if (attr_axis.size() > shape_.size()) {
      MS_LOG(EXCEPTION) << "invalid axis size: " << axis_.size();
    } else if (attr_axis.empty()) {
      axis_.push_back(shape_.size() - 1);
    } else {
      for (auto axis : attr_axis) {
        if (IntToSize(axis) >= (shape_.size())) {
          MS_LOG(EXCEPTION) << "axis value is oversize.";
        }
        axis < 0 ? axis_.push_back(axis + shape_.size()) : axis_.push_back(axis);
      }
    }
  } else if (axis_addr->isa<Int32Imm>()) {
    int axis = AnfAlgo::GetNodeAttr<int>(kernel_node, AXIS);
    if (axis >= 0 && IntToSize(axis) >= shape_.size()) {
      MS_LOG(EXCEPTION) << "axis value is oversize.";
    }
    axis < 0 ? axis_.push_back(axis + shape_.size()) : axis_.push_back(axis);
  } else {
    MS_LOG(EXCEPTION) << "Attribute axis type is invalid.";
  }
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (shape_[i] <= 0) {
      MS_LOG(EXCEPTION) << "shape value is invalid.";
    }
    left_dims_ *= shape_[i];
  }
  for (size_t i = 0; i < axis_.size(); ++i) {
    stride_ *= shape_[axis_[i]];
  }
  if (stride_ <= 0) {
    MS_LOG(EXCEPTION) << "stride_ must greater than zero.";
  }
  left_dims_ = left_dims_ / stride_;
}
bool ReduceCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspaces*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "input or output empty!";
  }
  size_t out_float_size = left_dims_ * sizeof(float);
  size_t in_float_size = stride_ * out_float_size;
  if (inputs[0]->size != in_float_size || outputs[0]->size != out_float_size) {
    MS_LOG(EXCEPTION) << "invalid input or output data size!";
  }
  auto input = reinterpret_cast<float *>(inputs[0]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  int size = inputs[0]->size / sizeof(float);
  std::vector<float> new_input(IntToSize(size), 0.0);
  std::vector<size_t> transpose_axis;
  for (size_t i = 0; i < shape_.size(); ++i) {
    bool insert = true;
    for (size_t j = 0; j < axis_.size(); ++j) {
      if (axis_[j] == i) {
        insert = false;
        break;
      }
    }
    if (insert) {
      transpose_axis.push_back(i);
    }
  }
  (void)transpose_axis.insert(transpose_axis.end(), axis_.begin(), axis_.end());
  Transpose(size, input, shape_, transpose_axis, SizeToInt(shape_.size()), &new_input[0]);
  if (reduce_type_ == kReduceTypeMax) {
    for (size_t i = 0; i < left_dims_; ++i) {
      float value = new_input[i * stride_];
      for (size_t k = 0; k < stride_; ++k) {
        if (value < new_input[i * stride_ + k]) {
          value = new_input[i * stride_ + k];
        }
      }
      output[i] = value;
    }
  } else {
    for (size_t i = 0; i < left_dims_; ++i) {
      float value = 0.0;
      for (size_t k = 0; k < stride_; ++k) {
        value += new_input[i * stride_ + k];
      }
      if (reduce_type_ == kReduceTypeMean) {
        output[i] = value / stride_;
      } else {
        output[i] = value;
      }
    }
  }
  return true;
}
void ReduceCPUKernel::Transpose(const int size, const float *input, const std::vector<size_t> &input_shape,
                                const std::vector<size_t> &input_axis, const int shape_size, float *output) {
  int pos_array[kMaxDim];
  int size_offset[kMaxDim];
  size_offset[0] = size / SizeToInt(input_shape[0]);
  for (int i = 1; i < shape_size; i++) {
    size_offset[i] = size_offset[i - 1] / SizeToInt(input_shape[i]);
  }
  for (int position = 0; position < size; position += 1) {
    int temp_position = position;
    pos_array[0] = temp_position / size_offset[0];
    for (int i = 1; i < shape_size; i++) {
      temp_position -= pos_array[i - 1] * size_offset[i - 1];
      pos_array[i] = temp_position / size_offset[i];
    }
    int new_position = pos_array[SizeToInt(input_axis[shape_size - 1])];
    int new_position_size = 1;
    for (int j = shape_size - 2; j >= 0; j--) {
      new_position_size *= SizeToInt(input_shape[SizeToInt(input_axis[j + 1])]);
      new_position += pos_array[SizeToInt(input_axis[j])] * new_position_size;
    }
    output[new_position] = input[position];
  }
  return;
}
}  // namespace kernel
}  // namespace mindspore
