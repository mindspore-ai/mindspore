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
#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"
#include "backend/kernel_compiler/cpu/dropout_grad_kernel.h"

namespace mindspore {
namespace kernel {
void DropoutGradCpuBwdKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);

  auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto input_mask_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (input_shape.size() != input_mask_shape.size()) {
    MS_LOG(EXCEPTION) << "Input size " << input_shape.size() << " and mask size " << input_mask_shape.size()
                      << " is not match";
  }
  num_count_ = 1;
  for (size_t x : input_shape) {
    num_count_ *= x;
  }
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  keep_prob_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "keep_prob");
  if (keep_prob_ == 0) {
    MS_LOG(EXCEPTION) << "The keep_prob is zero.";
  }
}

bool DropoutGradCpuBwdKernel::Launch(const std::vector<AddressPtr> &inputs,
                                     const std::vector<AddressPtr> & /*workspace*/,
                                     const std::vector<AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    DropoutBackwardKernel<float16>(inputs, outputs, num_count_, keep_prob_);
  } else if (dtype_ == kNumberTypeFloat32) {
    DropoutBackwardKernel<float>(inputs, outputs, num_count_, keep_prob_);
  }

  return true;
}

template <typename T>
void DropoutGradCpuBwdKernel::DropoutBackwardKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &outputs, size_t num_count,
                                                    float keep_prob) {
  auto dx = reinterpret_cast<T *>(outputs[0]->addr);
  auto dy = reinterpret_cast<T *>(inputs[0]->addr);
  auto mask = reinterpret_cast<T *>(inputs[1]->addr);

  const float scale = 1.f / keep_prob;
  for (size_t i = 0; i < num_count; i += 1) {
    dx[i] = (T)(scale * static_cast<float>(dy[i] * mask[i]));
  }
}
}  // namespace kernel
}  // namespace mindspore
