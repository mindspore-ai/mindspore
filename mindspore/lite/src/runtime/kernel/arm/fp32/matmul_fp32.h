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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_MATMUL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_MATMUL_H_

#include <vector>
#include "nnacl/matmul_parameter.h"
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class MatmulCPUKernel : public LiteKernel {
 public:
  explicit MatmulCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                           const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    params_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~MatmulCPUKernel() override;
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int Eval() override;

 private:
  int MallocMatrixABuffer();
  int MallocMatrixBBuffer();
  int InitBias();
  void InitMatrixA(const float *src_ptr, float *dst_ptr);
  void InitMatrixB(const float *src_ptr, float *dst_ptr);
  void FreeTmpBuffer();

 private:
  MatMulParameter *params_ = nullptr;
  float *a_pack_ptr_ = nullptr;
  float *b_pack_ptr_ = nullptr;
  float *bias_ptr_ = nullptr;
  float *a_ptr_ = nullptr;
  float *b_ptr_ = nullptr;
  float *cur_a_ptr_ = nullptr;
  float *cur_b_ptr_ = nullptr;
  float *cur_c_ptr_ = nullptr;
  bool is_vector_a_ = false;
  int col_tile_ = 0;
  int thread_stride_ = 0;
  int thread_count_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_MATMUL_H_
