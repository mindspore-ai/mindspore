/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/masked_select_grad_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaskedSelectGradInputsNum = 3;
constexpr size_t kMaskedSelectGradOutputsNum = 1;
}  // namespace

template <typename T>
void MaskedSelectGradCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shape_a_ = AnfAlgo::GetInputDeviceShape(kernel_node, INPUT);
  input_shape_b_ = AnfAlgo::GetInputDeviceShape(kernel_node, MASK);
  grad_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, GRAD);
  output_shape_ = CPUKernelUtils::GetBroadcastShape(input_shape_a_, input_shape_b_);
  for (const uint64_t &d : output_shape_) {
    tensor_size_ *= d;
  }
}

template <typename T>
bool MaskedSelectGradCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaskedSelectGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaskedSelectGradOutputsNum, kernel_name_);
  auto mask = reinterpret_cast<bool *>(inputs[MASK]->addr);
  auto grad = reinterpret_cast<T *>(inputs[GRAD]->addr);
  auto dx = reinterpret_cast<T *>(outputs[INPUT]->addr);

  auto ret = memset_s(dx, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Memset output[0] of kernel MaskedSelectGrad failed, ret: " << ret;
  }

  uint64_t output_size = outputs[0]->size / sizeof(T);
  uint64_t j = 0;
  if (input_shape_a_ == input_shape_b_) {
    for (uint64_t i = 0; i < output_size; ++i) {
      if (mask[i]) {
        dx[i] += grad[j++];
      }
    }
  } else {
    BroadcastIterator iter(input_shape_a_, input_shape_b_, output_shape_);
    iter.SetPos(0);
    for (uint64_t i = 0; i < tensor_size_; ++i) {
      if (mask[iter.GetInputPosB()]) {
        dx[iter.GetInputPosA()] += grad[j++];
      }
      iter.GenNextPos();
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
