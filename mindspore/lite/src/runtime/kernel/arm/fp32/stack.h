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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_STACK_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_STACK_H_

#include <vector>
#include "src/lite_kernel.h"

#include "src/runtime/kernel/arm/base/layout_transform.h"

namespace mindspore::kernel {
class StackCPUKernel : public LiteKernel {
 public:
  StackCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                 const std::vector<lite::tensor::Tensor *> &outputs)
      : LiteKernel(parameter, inputs, outputs),
        convert_functions_(inputs_.size(), nullptr),
        packed_inputs_(inputs_.size(), nullptr) {}

  ~StackCPUKernel() {
    for (size_t i = 0; i < packed_inputs_.size(); ++i) {
      if (packed_inputs_[i] != nullptr) {
        free(packed_inputs_[i]);
        packed_inputs_[i] = nullptr;
      }
    }
  }

  int Init() override;
  int ReSize() override { return 0; }
  int Run() override;

 private:
  int axis_;
  std::vector<LayoutConvertor> convert_functions_;
  std::vector<float *> packed_inputs_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_STACK_H_

