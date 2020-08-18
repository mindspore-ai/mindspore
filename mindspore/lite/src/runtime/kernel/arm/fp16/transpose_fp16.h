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

namespace mindspore::kernel {

class TransposeFp16CPUKernel : public LiteKernel {
 public:
  explicit TransposeFp16CPUKernel(OpParameter *param, const std::vector<lite::tensor::Tensor *> &inputs,
                                  const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                                  const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(param, inputs, outputs, ctx, primitive), thread_num_(ctx->thread_num_) {}
  ~TransposeFp16CPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override;
  int TransposeParallel(int task_id);

 private:
  int thread_num_;
  int thread_h_stride_;
  int thread_h_num_;
  int num_unit_;
  float *in_data_;
  float *out_data_;
  float16_t *fp16_in_data_ = nullptr;
  float16_t *fp16_out_data_ = nullptr;
  int *in_shape_;
  int *out_shape_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP16_TRANSPOSE_FP16_H_
