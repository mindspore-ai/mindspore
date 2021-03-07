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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_SMOOTH_L1_LOSS_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_SMOOTH_L1_LOSS_GRAD_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp32_grad/smooth_l1_loss.h"

namespace mindspore::kernel {
class SmoothL1LossGradCPUKernel : public LiteKernel {
 public:
  explicit SmoothL1LossGradCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), smooth_l1_param_(nullptr), thread_count_(ctx->thread_num_) {
    smooth_l1_param_ = reinterpret_cast<SmoothL1LossParameter *>(parameter);
  }
  ~SmoothL1LossGradCPUKernel() override {}
  int Init() override;
  int ReSize() override;
  int Run() override;
  int Execute(int task_id);

 private:
  SmoothL1LossParameter *smooth_l1_param_;
  int thread_count_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_GRAD_SMOOTH_L1_LOSS_GRAD_H_
