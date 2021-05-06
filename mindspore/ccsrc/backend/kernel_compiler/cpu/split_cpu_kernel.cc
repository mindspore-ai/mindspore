/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include "backend/kernel_compiler/cpu/split_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

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
  param_ = new SplitParameter();
  param_->num_split_ = output_num_;
  param_->split_dim_ = axis_ >= 0 ? axis_ : input_shape_.size() + axis_;

  param_->strides_[input_shape_.size() - 1] = 1;
  for (int i = input_shape_.size() - 2; i >= 0; i--) {
    param_->strides_[i] = param_->strides_[i + 1] * input_shape_[i + 1];
  }

  param_->split_sizes_ = new int[sizeof(int) * param_->num_split_];
  int split_size = input_shape_[param_->split_dim_] / output_num_;
  for (int i = 0; i < param_->num_split_; i++) {
    param_->split_sizes_[i] = split_size;
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
void SplitCPUKernel<T>::LaunchSplit(T *input, T **output, size_t size) {
  (void)std::transform(input_shape_.begin(), input_shape_.end(), std::back_inserter(input_shape_int_),
                       [](const int &value) { return static_cast<int>(value); });
  auto max_thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  const float block_size = 128.0;
  size_t thread_num = size < block_size * max_thread_num ? std::ceil(size / block_size) : max_thread_num;

  param_->split_count_ = size / (input_shape_[param_->split_dim_] * param_->strides_[param_->split_dim_]);
  int num_unit = param_->split_count_ * param_->num_split_;
  int thread_n_stride;
  if (thread_num != 0) {
    thread_n_stride = UP_DIV(num_unit, thread_num);
  }

  auto task = [&](size_t start, size_t end) {
    int task_id = start / (size / thread_num);
    int thread_offset = task_id * thread_n_stride;
    int num_unit_thread = MSMIN(thread_n_stride, num_unit - task_id * thread_n_stride);
    DoSplit(input, reinterpret_cast<void **>(output), &input_shape_int_[0], thread_offset, num_unit_thread, param_,
            sizeof(T));
  };
  CPUKernelUtils::ParallelFor(task, size);

  return;
}

template <typename T>
void SplitCPUKernel<T>::FreeTmpBuff() {
  if (param_->split_sizes_ != nullptr) {
    delete[] param_->split_sizes_;
    param_->split_sizes_ = nullptr;
  }
  if (param_ != nullptr) {
    delete param_;
    param_ = nullptr;
  }
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
  FreeTmpBuff();
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
