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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SPLIT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SPLIT_H_
#include <arm_neon.h>

#include <vector>
#include "src/runtime/kernel/arm/base/split_base.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class SplitFp16CPUKernel : public SplitBaseCPUKernel {
 public:
  SplitFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                     const mindspore::lite::PrimitiveC *primitive)
      : SplitBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~SplitFp16CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int Split(int task_id);

 private:
  float16_t *input_ptr_ = nullptr;
  std::vector<float16_t *> output_ptr_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_SPLIT_H_
