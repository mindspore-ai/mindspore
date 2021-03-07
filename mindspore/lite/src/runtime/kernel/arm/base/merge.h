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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_MERGE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_MERGE_H_

#include <vector>
#include "src/runtime/kernel/arm/base/carry_data.h"
#include "src/tensor.h"
#include "src/tensorlist.h"

namespace mindspore::kernel {
enum InputPart { UNKNOWN_INPUT_PART, LEFT_INPUT_PART, RIGHT_INPUT_PART };

class MergeCPUKernel : public CarryDataKernel {
 public:
  MergeCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : CarryDataKernel(parameter, inputs, outputs, ctx) {}
  bool IsReady(const std::vector<lite::Tensor *> &scope_tensors) override;
  ~MergeCPUKernel() override = default;
  int FreeInWorkTensor() const override;
  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  InputPart FindReadyPart(const std::vector<lite::Tensor *> &scope_tensors);

 private:
  InputPart ready_part_ = UNKNOWN_INPUT_PART;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_MERGE_H_
