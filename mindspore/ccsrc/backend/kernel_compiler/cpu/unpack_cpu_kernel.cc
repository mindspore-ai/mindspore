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

#include "backend/kernel_compiler/cpu/unpack_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void UnpackCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  int64_t axis_tmp = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "axis");
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (axis_tmp < 0) {
    axis_tmp += SizeToLong(input_shape.size());
  }
  size_t axis_ = LongToSize(axis_tmp);
  output_num_ = LongToSize(AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "num"));
  for (size_t i = 0; i < input_shape.size(); i++) {
    input_size_ *= input_shape[i];
    if (i > IntToSize(axis_)) {
      dims_after_axis_ *= input_shape[i];
    }
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

template <typename T>
void UnpackCPUKernel<T>::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  workspace_size_list_.emplace_back(sizeof(T *) * output_num_);
}

template <typename T>
bool UnpackCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                const std::vector<kernel::AddressPtr> &workspace,
                                const std::vector<kernel::AddressPtr> &outputs) {
  LaunchKernel(inputs, workspace, outputs);
  return true;
}

template <typename T>
void UnpackCPUKernel<T>::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  input_ = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_);
  outputs_host_ = reinterpret_cast<T **>(workspace[0]->addr);
  MS_EXCEPTION_IF_NULL(outputs_host_);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs_host_[i] = reinterpret_cast<T *>(outputs[i]->addr);
    MS_EXCEPTION_IF_NULL(outputs_host_[i]);
  }
  auto max_thread_num = std::thread::hardware_concurrency();
  size_t thread_num = input_size_ < 128 * max_thread_num ? std::ceil(input_size_ / 128.0) : max_thread_num;
  if (thread_num < 1) {
    MS_LOG(ERROR) << "Invalid value: thread_num" << thread_num;
    return;
  }
  std::vector<std::thread> threads;
  threads.reserve(thread_num);
  size_t start = 0;
  size_t one_gap = (input_size_ + thread_num - 1) / thread_num;
  if (one_gap < 1) {
    MS_LOG(ERROR) << "Invalid value: one_gap " << one_gap;
    return;
  }
  while (start < input_size_) {
    size_t end = (start + one_gap) > input_size_ ? input_size_ : (start + one_gap);
    threads.emplace_back(std::thread(&UnpackCPUKernel::UnpackResult, this, start, end));
    start += one_gap;
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

template <typename T>
void UnpackCPUKernel<T>::UnpackResult(const size_t start, const size_t end) {
  for (size_t i = start; i < end; ++i) {
    size_t output_index = (i / dims_after_axis_) % output_num_;
    size_t number_of_reset = output_num_ * dims_after_axis_;
    size_t tensor_index = i / number_of_reset * dims_after_axis_ + i % dims_after_axis_;
    outputs_host_[output_index][tensor_index] = input_[i];
  }
}

template <typename T>
void UnpackCPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but UnpackCPUKernel needs 1 input.";
  }
}
}  // namespace kernel
}  // namespace mindspore
