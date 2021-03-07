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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_ARITHMETIC_SELF_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_ARITHMETIC_SELF_GRAD_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp16_grad/arithmetic_self_grad.h"

namespace mindspore::kernel {
class ArithmeticSelfGradFp16CPUKernel : public LiteKernel {
 public:
  explicit ArithmeticSelfGradFp16CPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(param, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    param_act_grad_ = reinterpret_cast<ArithmeticSelfGradParameterFp16 *>(param);
  }
  ~ArithmeticSelfGradFp16CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoActivation(int task_id);

 private:
  ArithmeticSelfGradParameterFp16 *param_act_grad_;
  int thread_count_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GRAD_ARITHMETIC_SELF_GRAD_H_
