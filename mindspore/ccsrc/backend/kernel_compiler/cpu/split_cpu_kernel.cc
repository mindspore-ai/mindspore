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
#include "backend/kernel_compiler/cpu/split_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void SplitCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "axis");
  output_num_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "output_num");
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  CheckParam(kernel_node);
  Reshape();
}

template <typename T>
void SplitCPUKernel<T>::Reshape() {
  input_size_ = 1;
  dims_current_after_axis_ = 1;
  dims_after_axis_ = 1;
  axis_step_ = input_shape_[axis_] / output_num_;

  for (int i = 0; i < SizeToInt(input_shape_.size()); i++) {
    input_size_ *= input_shape_[i];
    if (i > axis_) {
      dims_current_after_axis_ *= input_shape_[i];
      dims_after_axis_ *= input_shape_[i];
    }
    if (i == axis_) {
      dims_current_after_axis_ *= input_shape_[i];
    }
  }
}

template <typename T>
void SplitCPUKernel<T>::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  workspace_size_list_.emplace_back((sizeof(T *) * output_num_));
}

template <typename T>
bool SplitCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &workspace,
                               const std::vector<kernel::AddressPtr> &outputs) {
  LaunchKernel(inputs, workspace, outputs);
  return true;
}

template <typename T>
void SplitCPUKernel<T>::LaunchSplit(const T *input, T **output, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int num = i % dims_current_after_axis_ / dims_after_axis_;
      int block = num / axis_step_;
      int block_pos = i / dims_current_after_axis_ * axis_step_ * dims_after_axis_ +
                      num % axis_step_ * dims_after_axis_ + i % dims_after_axis_;
      output[block][block_pos] = input[i];
    }
  };
  CPUKernelUtils::ParallelFor(task, size);
  return;
}

template <typename T>
void SplitCPUKernel<T>::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T **output = reinterpret_cast<T **>(workspace[0]->addr);
  for (size_t i = 0; i < outputs.size(); i++) {
    output[i] = reinterpret_cast<T *>(outputs[i]->addr);
  }
  size_t size = static_cast<size_t>(inputs[0]->size / sizeof(T));
  LaunchSplit(input, output, size);
  return;
}

template <typename T>
void SplitCPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  auto input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  int64_t dims = SizeToLong(input_shape_.size());
  int64_t output_num = SizeToLong(AnfAlgo::GetOutputTensorNum(kernel_node));

  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but Split needs 1 input.";
  }
  if (dims == 0) {
    MS_LOG(EXCEPTION) << "Input dims is " << dims << ", scalar is not supported.";
  }
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "Attr axis_ " << axis_ << " must be in " << -dims << "~" << dims;
  }
  if (axis_ < 0) {
    axis_ += SizeToInt(input_shape_.size());
  }
  if (output_num_ > SizeToInt(input_shape_[axis_])) {
    MS_LOG(EXCEPTION) << "Attr output_num " << output_num_ << " must less than " << input_shape_[axis_];
  }
  if (output_num_ != output_num) {
    MS_LOG(EXCEPTION) << "Output num is " << output_num << ", but need " << output_num_;
  }
}
}  // namespace kernel
}  // namespace mindspore
