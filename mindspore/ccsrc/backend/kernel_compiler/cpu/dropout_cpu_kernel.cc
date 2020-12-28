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

#include <algorithm>
#include <random>
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/cpu/dropout_cpu_kernel.h"

namespace mindspore {
namespace kernel {
void DropoutCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  mask_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 1);
  keep_prob_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "keep_prob");
  if (keep_prob_ <= 0.0) {
    MS_LOG(EXCEPTION) << "Keep_prob is smaller or equal to zero but DropoutCPUKernel needs greater than 0";
  }
  if (keep_prob_ > 1.0) {
    MS_LOG(EXCEPTION) << "Keep_prob greater than one but DropoutCPUKernel needs smaller or equal to one";
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  for (const uint64_t &d : input_shape_) {
    tensor_size_ *= d;
  }
}

bool DropoutCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                              const std::vector<kernel::AddressPtr> & /*workspace*/,
                              const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  }
  return true;
}

template <typename T>
void DropoutCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto mask_addr = reinterpret_cast<T *>(outputs[1]->addr);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dis(keep_prob_);
  T scale = (T)(1.f / keep_prob_);
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    mask_addr[i] = (T)dis(gen);
    output_addr[i] = mask_addr[i] * input_addr[i] * scale;
  }
}

void DropoutCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but DropoutCPUKernel needs 1 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 2) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but DropoutCPUKernel needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
