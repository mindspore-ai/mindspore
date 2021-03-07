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

#ifndef MINDSPORE_ACTIVATION_FP16_GRAD_H
#define MINDSPORE_ACTIVATION_FP16_GRAD_H

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp16_grad/activation_grad.h"

namespace mindspore::kernel {
class ActivationGradCPUKernelFp16 : public LiteKernel {
 public:
  explicit ActivationGradCPUKernelFp16(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(param, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    param_act_grad_ = reinterpret_cast<ActivationGradParameterFp16 *>(param);
  }
  ~ActivationGradCPUKernelFp16() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoActivation(int task_id);

 private:
  ActivationGradParameterFp16 *param_act_grad_;
  int thread_count_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_ACTIVATION_FP16_GRAD_H
