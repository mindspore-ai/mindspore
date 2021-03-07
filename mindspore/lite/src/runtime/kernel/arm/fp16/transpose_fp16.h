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

#ifndef MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP16_TRANSPOSE_FP16_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP16_TRANSPOSE_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/transpose_fp32.h"

namespace mindspore::kernel {

class TransposeFp16CPUKernel : public TransposeCPUKernel {
 public:
  explicit TransposeFp16CPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                                  const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : TransposeCPUKernel(param, inputs, outputs, ctx) {}
  ~TransposeFp16CPUKernel() = default;

  int Init() override;
  int Run() override;

 private:
  float16_t *in_data_fp16_ = nullptr;
  float16_t *out_data_fp16_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP16_TRANSPOSE_FP16_H_
