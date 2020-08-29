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
#include "src/lite_kernel.h"
#include "nnacl/matmul_parameter.h"
#include "src/runtime/kernel/arm/base/matmul_base.h"

namespace mindspore::kernel {
class MatmulCPUKernel : public MatmulBaseCPUKernel {
 public:
  explicit MatmulCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                           const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                           const mindspore::lite::PrimitiveC *primitive)
      : MatmulBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~MatmulCPUKernel() override;
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 private:
  void InitMatrixA(float *src_ptr, float *dst_ptr);
  void InitMatrixB(float *src_ptr, float *dst_ptr);
  void FreeTmpBuffer();

 private:
  float *a_c12_ptr_ = nullptr;
  float *b_r8_ptr_ = nullptr;
  float *bias_ptr_ = nullptr;
  float *a_ptr_ = nullptr;
  float *b_ptr_ = nullptr;
  float *c_ptr_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_MATMUL_H_
