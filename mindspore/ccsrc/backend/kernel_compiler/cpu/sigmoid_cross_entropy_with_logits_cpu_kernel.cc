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

#include "backend/kernel_compiler/cpu/sigmoid_cross_entropy_with_logits_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSigmoidCrossEntropyWithLogitsInputsNum = 2;
constexpr size_t kSigmoidCrossEntropyWithLogitsOutputsNum = 1;
}  // namespace

void SigmoidCrossEntropyWithLogitsCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  std::vector<size_t> x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (const uint64_t &d : x_shape) {
    tensor_size_ *= d;
  }
}

bool SigmoidCrossEntropyWithLogitsCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << dtype_;
  }
  return true;
}

template <typename T>
void SigmoidCrossEntropyWithLogitsCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                          const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSigmoidCrossEntropyWithLogitsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSigmoidCrossEntropyWithLogitsOutputsNum, kernel_name_);
  auto *logits_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *labels_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto zero = static_cast<T>(0.0);
  auto one = static_cast<T>(1.0);
  auto two = static_cast<T>(2.0);
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    if (logits_addr[i] >= zero) {
      output_addr[i] = static_cast<T>(log1p(static_cast<float>(exp(logits_addr[i] - two * logits_addr[i])))) -
                       logits_addr[i] * (labels_addr[i] - one);
    } else {
      output_addr[i] = static_cast<T>(log1p(static_cast<float>(exp(logits_addr[i])))) - logits_addr[i] * labels_addr[i];
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
