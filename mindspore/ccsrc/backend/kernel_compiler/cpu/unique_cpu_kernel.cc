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

#include "backend/kernel_compiler/cpu/unique_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void UniqueCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  n_ = input_shape[0];
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool UniqueCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  }
  return true;
}

template <typename T>
void UniqueCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto y_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto idx_addr = reinterpret_cast<int *>(outputs[1]->addr);

  std::unordered_map<T, int> uniq;
  int n = SizeToInt(n_);
  uniq.reserve(n * 2);
  for (int i = 0, j = 0; i < n; ++i) {
    auto it = uniq.emplace(x_addr[i], j);
    idx_addr[i] = it.first->second;
    if (it.second) {
      ++j;
    }
  }
  for (const auto &it : uniq) {
    y_addr[it.second] = it.first;
  }
}

void UniqueCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but UniqueCPUKernel only support 1d.";
  }
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but UniqueCPUKernel needs 1 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but UniqueCPUKernel needs 2 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
