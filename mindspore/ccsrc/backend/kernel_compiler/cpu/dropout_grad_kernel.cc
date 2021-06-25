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
#include "nnacl/fp32_grad/dropout_grad.h"

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
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  keep_prob_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "keep_prob");
  if (keep_prob_ == 0) {
    MS_LOG(EXCEPTION) << "The keep_prob is zero.";
  }
}

bool DropoutGradCpuBwdKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    DropoutBackwardKernel<float16>(inputs, outputs, num_count_, keep_prob_);
  } else if (dtype_ == kNumberTypeFloat32) {
    DropoutBackwardKernel<float>(inputs, outputs, num_count_, keep_prob_);
  } else {
    MS_LOG(ERROR) << "Input data type: " << dtype_ << " is not supported for DropoutGrad kernel for CPU.";
  }

  return true;
}

template <typename T>
void DropoutGradCpuBwdKernel::DropoutBackwardKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &outputs, size_t num_count,
                                                    float keep_prob) {
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *mask = reinterpret_cast<T *>(inputs[1]->addr);
  const float scale = 1.f / keep_prob;

  if constexpr (std::is_same_v<T, float16>) {
    float *input_tmp = new float[num_count_];
    float *output_tmp = new float[num_count_];
    float *mask_tmp = new float[num_count_];
    for (size_t i = 0; i < num_count_; ++i) {
      input_tmp[i] = static_cast<float>(input[i]);
      mask_tmp[i] = static_cast<float>(mask[i]);
    }
    DropoutGrad(input_tmp, mask_tmp, output_tmp, num_count_, scale);
    for (size_t i = 0; i < num_count_; ++i) {
      output[i] = static_cast<float16>(output_tmp[i]);
    }
    delete[] input_tmp;
    delete[] output_tmp;
    delete[] mask_tmp;
  } else if constexpr (std::is_same_v<T, float>) {
    DropoutGrad(input, mask, output, num_count_, scale);
  }
}
}  // namespace kernel
}  // namespace mindspore
