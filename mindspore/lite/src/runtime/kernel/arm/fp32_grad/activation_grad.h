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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ACTIVATION_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ACTIVATION_GRAD_H_

#include <vector>
#include "src/lite_kernel.h"
#include "ir/anf.h"

#include "src/runtime/kernel/arm/nnacl/activation_grad.h"

namespace mindspore::kernel {
class ActivationGradCPUKernel : public LiteKernel {
 public:
  explicit ActivationGradCPUKernel(OpParameter *param, const std::vector<lite::tensor::Tensor *> &inputs,
                                   const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                                   const lite::Primitive *primitive)
      : LiteKernel(param, inputs, outputs, ctx, primitive) {
    ActivationGradParameter *param_act_grad = reinterpret_cast<ActivationGradParameter *>(param);
    type_ = param_act_grad->type_;
    alpha_ = param_act_grad->alpha_;
  }
  ~ActivationGradCPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoActivation(int task_id);

 private:
  int thread_count_;
  int type_;
  float alpha_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_ACTIVATION_GRAD_H_
