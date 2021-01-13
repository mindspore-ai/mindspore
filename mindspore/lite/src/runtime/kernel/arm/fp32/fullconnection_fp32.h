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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FULLCONNECTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FULLCONNECTION_H_

#include <vector>
#include "include/context.h"
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/lite_kernel.h"

using mindspore::lite::InnerContext;
namespace mindspore::kernel {
class FullconnectionCPUKernel : public LiteKernel {
 public:
  FullconnectionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx,
                          const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {
    fc_param_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~FullconnectionCPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int DoMatmul(int task_id);
  void FreeBuf();

 private:
  void InitMatrixA(const float *src_ptr, float *dst_ptr);
  void InitMatrixB(const float *src_ptr, float *dst_ptr);

 private:
  MatMulParameter *fc_param_ = nullptr;
  float *a_pack_ptr_ = nullptr;
  float *b_pack_ptr_ = nullptr;
  float *c_ptr_ = nullptr;
  float *bias_ptr_ = nullptr;
  float *a_ptr_ = nullptr;
  float *b_ptr_ = nullptr;
  bool is_vector_input_ = false;
  int thread_count_ = 1;
  int thread_stride_ = 0;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_FULLCONNECTION_H_
