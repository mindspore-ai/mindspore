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

#include "backend/kernel_compiler/cpu/dropout_grad_kernel.h"

#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"

#include "backend/kernel_compiler/cpu/nnacl/fp32_grad/dropout_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDropoutGradInputsNum = 2;
constexpr size_t kDropoutGradOutputsNum = 1;
}  // namespace

void DropoutGradCpuBwdKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
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
  if (keep_prob_ <= 0.0 || keep_prob_ > 1.0) {
    MS_LOG(EXCEPTION) << kernel_name_ << "requires keep_prob should be in (0.0, 1.0], but got " << keep_prob_;
  }
}

void DropoutGradCpuBwdKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  if (dtype_ == kNumberTypeFloat16) {
    (void)workspace_size_list_.emplace_back(num_count_ * sizeof(float));
    (void)workspace_size_list_.emplace_back(num_count_ * sizeof(float));
    (void)workspace_size_list_.emplace_back(num_count_ * sizeof(float));
  }
}

bool DropoutGradCpuBwdKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDropoutGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDropoutGradOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    DropoutBackwardKernel<float16>(inputs, workspace, outputs, keep_prob_);
  } else if (dtype_ == kNumberTypeFloat32) {
    DropoutBackwardKernel<float>(inputs, workspace, outputs, keep_prob_);
  } else {
    MS_LOG(EXCEPTION) << kernel_name_ << " only support float16 and float32 on CPU, but got "
                      << TypeIdToType(dtype_)->ToString();
  }

  return true;
}

template <typename T>
void DropoutGradCpuBwdKernel::DropoutBackwardKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &workspace,
                                                    const std::vector<AddressPtr> &outputs, float keep_prob) {
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *mask = reinterpret_cast<T *>(inputs[1]->addr);
  const float scale = 1.f / keep_prob;

  if constexpr (std::is_same_v<T, float16>) {
    float *input_tmp = reinterpret_cast<float *>(workspace[0]->addr);
    float *output_tmp = reinterpret_cast<float *>(workspace[1]->addr);
    float *mask_tmp = reinterpret_cast<float *>(workspace[2]->addr);
    for (size_t i = 0; i < num_count_; ++i) {
      input_tmp[i] = static_cast<float>(input[i]);
      mask_tmp[i] = static_cast<float>(mask[i]);
    }
    DropoutGrad(input_tmp, mask_tmp, output_tmp, SizeToInt(num_count_), scale);
    for (size_t i = 0; i < num_count_; ++i) {
      output[i] = static_cast<float16>(output_tmp[i]);
    }
  } else if constexpr (std::is_same_v<T, float>) {
    DropoutGrad(input, mask, output, SizeToInt(num_count_), scale);
  }
}
}  // namespace kernel
}  // namespace mindspore
