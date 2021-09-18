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

#include "backend/kernel_compiler/cpu/dropout_cpu_kernel.h"

#include <algorithm>
#include <random>

#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDropoutInputsNum = 1;
constexpr size_t kDropoutOutputsNum = 2;
}  // namespace

void DropoutCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  mask_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 1);
  keep_prob_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "keep_prob");
  if (keep_prob_ <= 0.0 || keep_prob_ > 1.0) {
    MS_LOG(EXCEPTION) << kernel_name_ << "requires keep_prob should be in (0.0, 1.0], but got " << keep_prob_;
  }
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  for (const uint64_t &d : input_shape_) {
    tensor_size_ *= d;
  }
}

bool DropoutCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                              const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDropoutInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDropoutOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << kernel_name_ << " only support float16 and float32 on CPU, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

template <typename T>
void DropoutCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> &outputs) const {
  const auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
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
}  // namespace kernel
}  // namespace mindspore
