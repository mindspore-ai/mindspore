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

#include "backend/kernel_compiler/cpu/pack_cpu_kernel.h"
#include <thread>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPackOutputsNum = 1;
}  // namespace

template <typename T>
void PackCpuFwdKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_num_ = AnfAlgo::GetInputTensorNum(kernel_node);
  axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  if (axis_ < 0) {
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    axis_ += (SizeToInt(input_shape.size()) + 1);
  }

  // calculate elements while dim >= axis
  auto first_input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  for (size_t i = IntToSize(axis_); i < first_input_shape.size(); i++) {
    dims_behind_axis_ *= first_input_shape[i];
  }

  auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  output_size_ = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_size_ *= output_shape[i];
  }
}

template <typename T>
bool PackCpuFwdKernel<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPackOutputsNum, kernel_name_);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  inputs_host_ = std::make_unique<T *[]>(input_num_);
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs_host_[i] = reinterpret_cast<T *>(inputs[i]->addr);
  }

  // multi-threading
  size_t input_size = output_size_;
  size_t max_thread_num = std::max(std::thread::hardware_concurrency(), static_cast<unsigned int>(1));
  size_t use_thread_num =
    input_size < 128 * max_thread_num ? std::ceil(static_cast<float>(input_size / 128.0)) : max_thread_num;
  std::vector<std::thread> threads;

  if (use_thread_num < 1) {
    use_thread_num = 1;
  }

  threads.reserve(use_thread_num);
  size_t start = 0;
  size_t batch_size = (input_size + use_thread_num - 1) / use_thread_num;

  while (start < input_size) {
    size_t end = (start + batch_size) > input_size ? input_size : (start + batch_size);
    (void)threads.emplace_back(std::thread(&PackCpuFwdKernel::PackTensor, this, output, start, end));
    start += batch_size;
  }

  for (auto &it : threads) {
    it.join();
  }
  return true;
}

template <typename T>
void PackCpuFwdKernel<T>::PackTensor(T *output, size_t start, size_t end) const {
  for (size_t pos = start; pos < end; ++pos) {
    size_t cur_input_index = pos / dims_behind_axis_ % input_num_;
    size_t cycle_len = input_num_ * dims_behind_axis_;
    size_t local_index = pos / cycle_len * dims_behind_axis_ + pos % cycle_len % dims_behind_axis_;
    output[pos] = inputs_host_[cur_input_index][local_index];
  }
}
}  // namespace kernel
}  // namespace mindspore
