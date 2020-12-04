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

#include "backend/kernel_compiler/cpu/smooth_l1_loss_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void SmoothL1LossCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  beta_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "beta");
  CheckParam(kernel_node);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  std::vector<size_t> x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (const uint64_t &d : x_shape) {
    tensor_size_ *= d;
  }
}

bool SmoothL1LossCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
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
void SmoothL1LossCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  auto predict_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto target_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto result_addr = reinterpret_cast<T *>(outputs[0]->addr);
  T zero = (T)0.0;
  T half = (T)0.5;
  T beta = (T)beta_;
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    T diff = predict_addr[i] - target_addr[i];
    if (diff < zero) {
      diff = -diff;
    }
    if (diff < beta) {
      result_addr[i] = half * diff * diff / beta;
    } else {
      result_addr[i] = diff - (half * beta);
    }
  }
}

void SmoothL1LossCPUKernel::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 2) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but SmoothL1LossCPUKernel needs 2 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but SmoothL1LossCPUKernel needs 1 output.";
  }
  if (beta_ == 0.0) {
    MS_LOG(EXCEPTION) << "Attr beta can not be zero.";
  }
}
}  // namespace kernel
}  // namespace mindspore
