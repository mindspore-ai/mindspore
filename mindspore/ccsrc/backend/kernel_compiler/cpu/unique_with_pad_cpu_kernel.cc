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

#include "backend/kernel_compiler/cpu/unique_with_pad_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void UniqueWithPadCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  n_ = SizeToLong(input_shape[0]);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool UniqueWithPadCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> & /*workspace*/,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Only unsupported int32 or int64 dtype";
  }
  return true;
}

template <typename T>
void UniqueWithPadCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) {
  T *a = reinterpret_cast<T *>(inputs[0]->addr);
  T pad_num = *reinterpret_cast<T *>(inputs[1]->addr);
  T *out = reinterpret_cast<T *>(outputs[0]->addr);
  T *idx_vec = reinterpret_cast<T *>(outputs[1]->addr);

  for (int64_t i = 0; i < n_; ++i) {
    out[i] = pad_num;
  }
  std::unordered_map<T, int> uniq;
  uniq.reserve(n_);
  for (int64_t i = 0, j = 0; i < n_; ++i) {
    auto it = uniq.emplace(a[i], j);
    idx_vec[i] = it.first->second;
    if (it.second) {
      ++j;
    }
  }
  for (const auto &it : uniq) {
    out[it.second] = it.first;
  }
}

void UniqueWithPadCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but UniqueCPUKernel only support 1d.";
  }
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but UniqueCPUKernel needs 2 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but UniqueCPUKernel needs 2 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
