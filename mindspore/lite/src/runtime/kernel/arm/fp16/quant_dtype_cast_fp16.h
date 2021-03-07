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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_QUANTDTYPECAST_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_QUANTDTYPECAST_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class QuantDTypeCastFp16CPUKernel : public LiteKernel {
 public:
  QuantDTypeCastFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_num_(ctx->thread_num_) {}
  ~QuantDTypeCastFp16CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int QuantDTypeCast(int task_id);

 private:
  int thread_num_;
  int thread_n_num_;
  int thread_n_stride_;
  int num_unit_;
  int8_t *int8_ptr_;
  uint8_t *uint8_ptr_;
  float16_t *float16_ptr_;
  bool int_to_float_;
  bool is_uint8_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_QUANTDTYPECAST_FP16_H_
