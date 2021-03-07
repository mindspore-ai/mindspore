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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ACTIVATION_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ACTIVATION_FP16_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp16/activation_fp16.h"

namespace mindspore::kernel {
class ActivationFp16CPUKernel : public LiteKernel {
 public:
  ActivationFp16CPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(param, inputs, outputs, ctx), thread_count_(ctx->thread_num_) {
    type_ = (reinterpret_cast<ActivationParameter *>(param))->type_;
    alpha_ = (float16_t)((reinterpret_cast<ActivationParameter *>(param))->alpha_);
    min_val_ = (reinterpret_cast<ActivationParameter *>(param))->min_val_;
    max_val_ = (reinterpret_cast<ActivationParameter *>(param))->max_val_;
  }
  ~ActivationFp16CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int DoActivation(int task_id);

 private:
  int thread_count_;
  int type_;
  float16_t alpha_;
  float min_val_;
  float max_val_;
  float16_t *fp16_input_ = nullptr;
  float16_t *fp16_output_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ACTIVATION_FP16_H_
