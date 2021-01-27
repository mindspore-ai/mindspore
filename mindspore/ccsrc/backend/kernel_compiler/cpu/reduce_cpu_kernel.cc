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

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include "backend/kernel_compiler/cpu/reduce_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
const size_t kReduceTypeMax = 1;
const size_t kReduceTypeMean = 2;
const size_t kReduceTypeSum = 3;
const size_t kReduceTypeMin = 4;
const size_t kMaxDim = 100;
static std::map<std::string, int> reduce_types_map_ = {
  {"ReduceMax", 1}, {"ReduceMean", 2}, {"ReduceSum", 3}, {"ReduceMin", 4}};

template <typename T>
void ReduceCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);

  reduce_type_ = reduce_types_map_[kernel_name];
  if (reduce_type_ == 0) {
    MS_LOG(EXCEPTION) << "Array reduce kernel type " << kernel_name << " is not supported.";
  }
  shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  CheckAxis(kernel_node);
  if (shape_.empty()) {
    shape_.push_back(1);
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

template <typename T>
bool ReduceCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> & /*workspaces*/,
                                const std::vector<kernel::AddressPtr> &outputs) {
  size_t out_size = left_dims_ * sizeof(T);
  size_t in_size = stride_ * out_size;
  if (inputs[0]->size != in_size || outputs[0]->size != out_size) {
    MS_LOG(EXCEPTION) << "invalid input or output data size!";
  }
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  int size = inputs[0]->size / sizeof(T);
  std::vector<T> new_input(IntToSize(size), 0.0);
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
  ConvertDataToOutput(&new_input[0], output);
  return true;
}

template <typename T>
void ReduceCPUKernel<T>::CheckAxis(const CNodePtr &kernel_node) {
  auto axis_addr = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr(AXIS);
  if (axis_addr->isa<ValueTuple>() || axis_addr->isa<ValueList>()) {
    std::vector<int> attr_axis;
    std::vector<int64_t> attr_axis_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, AXIS);
    (void)std::transform(attr_axis_me.begin(), attr_axis_me.end(), std::back_inserter(attr_axis),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (attr_axis.size() > shape_.size()) {
      MS_LOG(EXCEPTION) << "invalid axis size: " << axis_.size();
    } else if (attr_axis.empty()) {
      for (size_t i = 0; i < shape_.size(); ++i) {
        axis_.push_back(i);
      }
    } else {
      for (auto axis : attr_axis) {
        while (axis < 0) {
          axis += SizeToInt(shape_.size());
        }
        if (IntToSize(axis) >= (shape_.size())) {
          MS_LOG(EXCEPTION) << "axis value is oversize.";
        }
        axis_.push_back(IntToSize(axis));
      }
    }
  } else if (axis_addr->isa<Int64Imm>()) {
    int axis = static_cast<int64_t>(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
    while (axis < 0) {
      axis += SizeToInt(shape_.size());
    }
    if (IntToSize(axis) >= shape_.size()) {
      MS_LOG(EXCEPTION) << "axis value is oversize.";
    }
    axis_.push_back(IntToSize(axis));
  } else {
    MS_LOG(EXCEPTION) << "Attribute axis type is invalid.";
  }
}

template <typename T>
void ReduceCPUKernel<T>::ConvertDataToOutput(const T *new_input, T *output) {
  if (reduce_type_ == kReduceTypeMax || reduce_type_ == kReduceTypeMin) {
    for (size_t i = 0; i < left_dims_; ++i) {
      T value = new_input[i * stride_];
      for (size_t k = 0; k < stride_; ++k) {
        if (reduce_type_ == kReduceTypeMax) {
          if (value < new_input[i * stride_ + k]) {
            value = new_input[i * stride_ + k];
          }
        } else {
          if (value > new_input[i * stride_ + k]) {
            value = new_input[i * stride_ + k];
          }
        }
      }
      output[i] = value;
    }
  } else if (reduce_type_ == kReduceTypeMean || reduce_type_ == kReduceTypeSum) {
    for (size_t i = 0; i < left_dims_; ++i) {
      T value = 0.0;
      for (size_t k = 0; k < stride_; ++k) {
        value += new_input[i * stride_ + k];
      }
      if (reduce_type_ == kReduceTypeMean) {
        output[i] = value / stride_;
      } else {
        output[i] = value;
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "Array reduce kernel type " << reduce_type_ << " is not supported.";
  }
}

template <typename T>
void ReduceCPUKernel<T>::Transpose(const int size, const T *input, const std::vector<size_t> &input_shape,
                                   const std::vector<size_t> &input_axis, const int shape_size, T *output) {
  int size_offset[kMaxDim];
  size_offset[0] = size / SizeToInt(input_shape[0]);
  for (int i = 1; i < shape_size; ++i) {
    size_offset[i] = size_offset[i - 1] / SizeToInt(input_shape[i]);
  }
  auto task = [&](size_t start, size_t end) {
    int pos_array[kMaxDim];
    for (size_t position = start; position < end; position += 1) {
      size_t temp_position = position;
      pos_array[0] = temp_position / size_offset[0];
      for (int i = 1; i < shape_size; ++i) {
        temp_position -= pos_array[i - 1] * size_offset[i - 1];
        pos_array[i] = temp_position / size_offset[i];
      }
      size_t new_position = pos_array[SizeToInt(input_axis[shape_size - 1])];
      size_t new_position_size = 1;
      for (int j = shape_size - 2; j >= 0; j--) {
        new_position_size *= SizeToInt(input_shape[SizeToInt(input_axis[j + 1])]);
        new_position += pos_array[SizeToInt(input_axis[j])] * new_position_size;
      }
      output[new_position] = input[position];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
  return;
}
}  // namespace kernel
}  // namespace mindspore
