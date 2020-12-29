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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_INT8_H_

#include <vector>
#include "include/context.h"
#include "nnacl/matmul_parameter.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "src/lite_kernel.h"

using mindspore::lite::InnerContext;
namespace mindspore::kernel {
class MatmulInt8CPUKernel : public LiteKernel {
 public:
  MatmulInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx,
                      const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    params_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~MatmulInt8CPUKernel() override;
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 private:
  void FreeTmpBuffer();

 private:
  MatMulParameter *params_ = nullptr;
  MatmulQuantArg quant_params_;
  int8_t *a_r4x16_ptr_ = nullptr;
  int8_t *b_c16x4_ptr_ = nullptr;
  int8_t *c_ptr_ = nullptr;
  int8_t *b_c16x4_batch_ = nullptr;
  int *bias_ptr_ = nullptr;
  int *input_sums_ = nullptr;
  int *weight_bias_sums_ = nullptr;
  int *weight_bias_sums_batch_ = nullptr;
  int thread_stride_ = 0;
  int thread_count_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_INT8_H_
