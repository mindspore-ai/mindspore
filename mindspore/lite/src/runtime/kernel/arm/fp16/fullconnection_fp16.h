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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FULLCONNECTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FULLCONNECTION_H_

#include <arm_neon.h>
#include <vector>
#include "include/errorcode.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/fp16/matmul_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class FullconnectionFP16CPUKernel : public LiteKernel {
 public:
  explicit FullconnectionFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                       const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    fc_param_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~FullconnectionFP16CPUKernel() override;
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 private:
  void InitMatrixA(float *a_ptr, float16_t *a_pack_ptr);
  void InitMatrixA(float16_t *a_ptr, float16_t *a_pack_ptr);
  void InitMatrixB(float *b_ptr, float16_t *b_pack_ptr);
  void InitMatrixB(float16_t *b_ptr, float16_t *b_pack_ptr);
  void FreeTmpBuffer();

 private:
  MatMulParameter *fc_param_ = nullptr;
  float16_t *a_pack_ptr_ = nullptr;
  float16_t *b_pack_ptr_ = nullptr;
  float16_t *bias_ptr_ = nullptr;
  float16_t *output_fp16_ = nullptr;
  float16_t *output_ptr_ = nullptr;
  float16_t *a_ptr_ = nullptr;
  float16_t *b_ptr_ = nullptr;
  bool is_vector_input_ = false;
  int thread_count_ = 1;
  int thread_stride_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FULLCONNECTION_H_
