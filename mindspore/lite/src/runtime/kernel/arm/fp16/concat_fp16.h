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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONCAT_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONCAT_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/base/concat_base.h"

using mindspore::lite::Context;

namespace mindspore::kernel {
class ConcatFp16CPUKernel : public ConcatBaseCPUKernel {
 public:
  ConcatFp16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                      const lite::Primitive *primitive)
      : ConcatBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}

  ~ConcatFp16CPUKernel() = default;

  int Init() override;

  int ReSize() override;

  int Run() override;

 private:
  void FreeTmpBuffer();

 private:
  std::vector<float16_t *> fp16_inputs_;
  float16_t *fp16_output_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONCAT_FP16_H_
