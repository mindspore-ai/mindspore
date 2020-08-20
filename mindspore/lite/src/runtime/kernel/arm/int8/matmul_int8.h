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
#include "src/runtime/kernel/arm/nnacl/quantization/quantize.h"
#include "src/runtime/kernel/arm/base/matmul_base.h"

using mindspore::lite::Context;

namespace mindspore::kernel {
class MatmulInt8CPUKernel : public MatmulBaseCPUKernel {
 public:
  MatmulInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                      const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                      const mindspore::lite::PrimitiveC *primitive)
      : MatmulBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~MatmulInt8CPUKernel() override;
  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 private:
  void FreeTmpBuffer() {
#ifdef ENABLE_ARM64
    if (a_r4d16_ptr_ != nullptr) {
      ctx_->allocator->Free(a_r4d16_ptr_);
      a_r4d16_ptr_ = nullptr;
    }
    if (b_c4d16_ptr_ != nullptr) {
      ctx_->allocator->Free(b_c4d16_ptr_);
      b_c4d16_ptr_ = nullptr;
    }
    if (c_r4c4_ptr_ != nullptr) {
      ctx_->allocator->Free(c_r4c4_ptr_);
      c_r4c4_ptr_ = nullptr;
    }
    if (a_sums_ != nullptr) {
      ctx_->allocator->Free(a_sums_);
      a_sums_ = nullptr;
    }
    if (b_bias_ != nullptr) {
      ctx_->allocator->Free(b_bias_);
      b_bias_ = nullptr;
    }
#else
    if (a_c8_ptr_ != nullptr) {
      ctx_->allocator->Free(a_c8_ptr_);
      a_c8_ptr_ = nullptr;
    }
    if (b_r8_ptr_ != nullptr) {
      ctx_->allocator->Free(b_r8_ptr_);
      b_r8_ptr_ = nullptr;
    }
    if (c_r8x8_ptr_ != nullptr) {
      ctx_->allocator->Free(c_r8x8_ptr_);
      c_r8x8_ptr_ = nullptr;
    }
#endif
  }
  MatmulQuantArg quant_params_;
#ifdef ENABLE_ARM64
  int8_t *a_r4d16_ptr_ = nullptr;
  int8_t *b_c4d16_ptr_ = nullptr;
  int8_t *c_r4c4_ptr_ = nullptr;
  int *a_sums_ = nullptr;
  int *b_bias_ = nullptr;
  int r4_;
  int c4_;
  int d16_;
#else
  int8_t *a_c8_ptr_ = nullptr;
  int8_t *b_r8_ptr_ = nullptr;
  int *c_r8x8_ptr_ = nullptr;
#endif
};  // namespace mindspore::kernel
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_MATMUL_INT8_H_
