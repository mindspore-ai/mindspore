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

#include "backend/kernel_compiler/cpu/split_cpu_kernel.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSplitInputsNum = 1;
}  // namespace

template <typename T>
void SplitCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  output_num_ = LongToSize(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "output_num"));
  if (output_num_ == 0) {
    MS_LOG(EXCEPTION) << "Attr output_num is equal to 0";
  }
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_),
                       [](const size_t &value) { return SizeToInt(value); });
  if (input_shape_.size() < 1 || input_shape_.size() > SPLIT_STRIDES_SIZE) {
    MS_LOG(EXCEPTION) << "Inpu shape size should not less than 1 or greater than " << SPLIT_STRIDES_SIZE << ", but got "
                      << input_shape_.size();
  }
  CheckParam(kernel_node);
}

template <typename T>
void SplitCPUKernel<T>::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  (void)workspace_size_list_.emplace_back((sizeof(T *) * LongToSize(output_num_)));
}

template <typename T>
bool SplitCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &workspace,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSplitInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num_, kernel_name_);
  LaunchKernel(inputs, workspace, outputs);
  return true;
}

template <typename T>
void SplitCPUKernel<T>::LaunchSplit(T *input, T **output, size_t /* size */) {
  SplitParameter param;
  param.num_split_ = SizeToInt(output_num_);
  param.split_dim_ = LongToInt(axis_);
  param.strides_[input_shape_.size() - 1] = 1;
  for (int i = SizeToInt(input_shape_.size()) - 2; i >= 0; i--) {  // from -2 to 0 dim
    param.strides_[i] = param.strides_[i + 1] * input_shape_[i + 1];
  }
  auto split_sizes = std::make_unique<int[]>(IntToSize(param.num_split_));
  param.split_sizes_ = split_sizes.get();
  int split_size = input_shape_[param.split_dim_] / SizeToInt(output_num_);
  for (int i = 0; i < param.num_split_; i++) {
    param.split_sizes_[i] = split_size;
  }
  param.split_count_ = 1;
  for (size_t i = 0; i < static_cast<size_t>(axis_); ++i) {
    param.split_count_ *= input_shape_[i];
  }
  auto task = [this, &input, &output, &param](size_t start, size_t end) {
    (void)DoSplit(input, reinterpret_cast<void **>(output), &input_shape_[0], SizeToInt(start), SizeToInt(end - start),
                  &param, SizeToInt(sizeof(T)));
  };
  ParallelLaunchAutoSearch(task, IntToSize(param.split_count_ * param.num_split_), this, &parallel_search_info_);
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
  int64_t dims = SizeToLong(input_shape_.size());
  if (dims == 0 || dims > SPLIT_STRIDES_SIZE) {
    MS_LOG(EXCEPTION) << "Input dims is " << dims << ", scalar is not supported.";
  }
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "Attr axis_ " << axis_ << " must be in " << -dims << "~" << dims;
  }
  if (axis_ < 0) {
    axis_ += SizeToLong(input_shape_.size());
  }
  if (output_num_ > IntToSize(input_shape_[LongToUlong(axis_)])) {
    MS_LOG(EXCEPTION) << "Attr output_num " << output_num_ << " must less than " << input_shape_[axis_];
  }
}
}  // namespace kernel
}  // namespace mindspore
