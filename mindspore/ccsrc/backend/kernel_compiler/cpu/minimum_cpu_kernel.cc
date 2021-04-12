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

#include "backend/kernel_compiler/cpu/minimum_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void MinimumCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  input_x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_y_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  TypeId input_x_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  TypeId input_y_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  size_t max_input_shape_size =
    input_x_shape_.size() > input_y_shape_.size() ? input_x_shape_.size() : input_y_shape_.size();
  for (size_t i = 0; i < output_shape_.size(); i++) {
    output_num_ *= output_shape_[i];
  }
  if ((input_x_shape_.size() == 0 && input_y_shape_.size() != 0) ||
      (input_x_shape_.size() != 0 && input_y_shape_.size() == 0)) {
    InitInputTensorAndScalar(max_input_shape_size);
  } else if (max_input_shape_size == output_shape_.size() && output_shape_.size() != 0) {
    InitInputTensors(input_x_dtype, input_y_dtype);
  } else {
    MS_LOG(EXCEPTION) << "Only support input two tensors or one tensor and one scalar";
  }
}

template <typename T>
void MinimumCPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but MinimumCPUKernel needs 2 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but MinimumCPUKernel needs 1 output.";
  }
}

template <typename T>
void MinimumCPUKernel<T>::InitInputTensorAndScalar(size_t max_input_shape_size) {
  if (max_input_shape_size != output_shape_.size()) {
    MS_LOG(EXCEPTION) << "Output tensor size must be equal to the max shape size of inputs";
  }
  need_broadcast_ = false;
}

template <typename T>
void MinimumCPUKernel<T>::InitInputTensors(TypeId input_x_dtype, TypeId input_y_dtype) {
  if (input_x_dtype == kNumberTypeBool && input_y_dtype == kNumberTypeBool) {
    MS_LOG(EXCEPTION) << "Input tensor types cannot be both bool";
  }
  // Check if the shape needs to be broadcast
  need_broadcast_ = IsBroadcast();
  if (need_broadcast_) {
    InitTensorBroadcastShape();
  }
}

template <typename T>
bool MinimumCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> & /*workspace*/,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  T *input_x_ = reinterpret_cast<T *>(inputs[0]->addr);
  T *input_y_ = reinterpret_cast<T *>(inputs[1]->addr);
  T *output_ = reinterpret_cast<T *>(outputs[0]->addr);
  BroadcastArith(input_x_, input_y_, output_);
  return true;
}

template <typename T>
void MinimumCPUKernel<T>::BroadcastArith(const T *input_x, const T *input_y, T *output) {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(output);
  if (need_broadcast_) {
    BroadcastArithKernel(broadcast_input_x_shape_[0], broadcast_input_x_shape_[1], broadcast_input_x_shape_[2],
                         broadcast_input_x_shape_[3], broadcast_input_x_shape_[4], broadcast_input_x_shape_[5],
                         broadcast_input_x_shape_[6], broadcast_input_y_shape_[0], broadcast_input_y_shape_[1],
                         broadcast_input_y_shape_[2], broadcast_input_y_shape_[3], broadcast_input_y_shape_[4],
                         broadcast_input_y_shape_[5], broadcast_input_y_shape_[6], broadcast_output_shape_[0],
                         broadcast_output_shape_[1], broadcast_output_shape_[2], broadcast_output_shape_[3],
                         broadcast_output_shape_[4], broadcast_output_shape_[5], broadcast_output_shape_[6], input_x,
                         input_y, output);
  } else {
    if (input_x_shape_.size() == 0 || input_y_shape_.size() == 0) {
      BroadcastArithOneScalarOneTensor(input_x, input_y, output);
    } else {
      BroadcastArithTensors(input_x, input_y, output);
    }
  }
}

template <typename T>
bool MinimumCPUKernel<T>::IsBroadcast() {
  if (input_x_shape_.size() != input_y_shape_.size()) {
    return true;
  }
  for (size_t i = 0; i < input_x_shape_.size(); i++) {
    if (input_x_shape_[i] != input_y_shape_[i]) {
      return true;
    }
  }
  return false;
}

template <typename T>
void MinimumCPUKernel<T>::InitTensorBroadcastShape() {
  if (output_shape_.size() > max_dims) {
    MS_LOG(EXCEPTION) << "Broadcast operation not support dim greater than 7";
  }
  broadcast_input_x_shape_.resize(max_dims, 1);
  broadcast_input_y_shape_.resize(max_dims, 1);
  broadcast_output_shape_.resize(max_dims, 1);
  for (size_t i = 0; i < output_shape_.size(); i++) {
    broadcast_output_shape_[i] = output_shape_[i];
  }
  int input_x_dim_offset = output_shape_.size() - input_x_shape_.size();
  for (size_t j = 0; j < input_x_shape_.size(); j++) {
    broadcast_input_x_shape_[j + input_x_dim_offset] = input_x_shape_[j];
    input_x_num_ *= input_x_shape_[j];
  }
  int input_y_dim_offset = output_shape_.size() - input_y_shape_.size();
  for (size_t k = 0; k < input_y_shape_.size(); k++) {
    if (need_broadcast_) {
      broadcast_input_y_shape_[k + input_y_dim_offset] = input_y_shape_[k];
      input_y_num_ *= input_y_shape_[k];
    }
  }
}

// Broadcast comparison
template <typename T>
size_t MinimumCPUKernel<T>::Index(const size_t &index, const size_t &dim) {
  return dim == 1 ? 0 : index;
}

// Broadcast Arithmetic
template <typename T>
void MinimumCPUKernel<T>::BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                               const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                               const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                               const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                               const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                               const size_t d6, const T *input_x, const T *input_y, T *output) {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(output);
  for (size_t pos = 0; pos < output_num_; pos++) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] = MinimumFunc(input_x[l_index], input_y[r_index]);
  }
}

template <typename T>
void MinimumCPUKernel<T>::BroadcastArithOneScalarOneTensor(const T *input_x, const T *input_y, T *output) {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(output);
  if (input_x_shape_.size() == 0) {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = MinimumFunc(input_x[0], input_y[i]);
    }
  } else {
    for (size_t i = 0; i < output_num_; ++i) {
      output[i] = MinimumFunc(input_x[i], input_y[0]);
    }
  }
}

template <typename T>
void MinimumCPUKernel<T>::BroadcastArithTensors(const T *input_x, const T *input_y, T *output) {
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(output);
  for (size_t i = 0; i < output_num_; ++i) {
    output[i] = MinimumFunc(input_x[i], input_y[i]);
  }
}
}  // namespace kernel
}  // namespace mindspore
