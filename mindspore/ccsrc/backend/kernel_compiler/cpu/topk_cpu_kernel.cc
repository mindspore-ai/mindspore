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

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include "backend/kernel_compiler/cpu/topk_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void TopKCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 2 || outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "TopK needs 2 inputs and 2 outputs, but get inputs: " << inputs.size()
                      << "outputs: " << outputs.size();
  }
  if (inputs[0]->size != outer_size_ * inner_size_ * sizeof(T)) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  if (inputs[1]->size != sizeof(int)) {
    MS_LOG(EXCEPTION) << "Input K must be int!";
  }
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  int k = reinterpret_cast<int *>(inputs[1]->addr)[0];
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto indices = reinterpret_cast<int *>(outputs[1]->addr);
  if (k < 1) {
    MS_LOG(EXCEPTION) << "Input k must > 0!";
  }
  int k_num = std::min<int>(inner_size_, k);
  if (outputs[0]->size != outer_size_ * k_num * sizeof(T)) {
    MS_LOG(EXCEPTION) << "Error output data size!";
  }
  for (size_t i = 0; i < outer_size_; ++i) {
    std::vector<size_t> idx(inner_size_);
    auto base_input = i * inner_size_;
    std::iota(idx.begin(), idx.end(), base_input);
    std::sort(idx.begin(), idx.end(),
              [&input](size_t index_1, size_t index_2) { return input[index_1] > input[index_2]; });
    auto base_output = i * k_num;
    if (!sorted_) {
      std::sort(idx.begin(), idx.begin() + k_num);
    }
    for (int j = 0; j < k_num; ++j) {
      indices[base_output + j] = idx[j] - base_input;
      output[base_output + j] = input[idx[j]];
    }
  }
}

void TopKCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (size_t i = 0; i < x_shape_.size() - 1; ++i) {
    outer_size_ *= x_shape_[i];
  }
  inner_size_ = x_shape_[x_shape_.size() - 1];
  sorted_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "sorted");
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool TopKCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                           const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
