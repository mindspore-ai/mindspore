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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_FULLCONNECTION_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_FULLCONNECTION_INT8_H_

#include <vector>
#include "src/runtime/kernel/arm/base/fullconnection_base.h"
#include "include/context.h"
#include "nnacl/quantization/quantize.h"
#include "nnacl/int8/common_func_int8.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class FullconnectionInt8CPUKernel : public FullconnectionBaseCPUKernel {
 public:
  FullconnectionInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx,
                              const mindspore::lite::PrimitiveC *primitive)
      : FullconnectionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~FullconnectionInt8CPUKernel() override { FreeTmpBuffer(); }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);

 private:
  void FreeTmpBuffer() {
    if (a_r4x16_ptr_ != nullptr) {
      ctx_->allocator->Free(a_r4x16_ptr_);
      a_r4x16_ptr_ = nullptr;
    }
    if (b_c16x4_ptr_ != nullptr) {
      ctx_->allocator->Free(b_c16x4_ptr_);
      b_c16x4_ptr_ = nullptr;
    }
    if (input_sums_ != nullptr) {
      ctx_->allocator->Free(input_sums_);
      input_sums_ = nullptr;
    }
    if (weight_bias_sums_ != nullptr) {
      ctx_->allocator->Free(weight_bias_sums_);
      weight_bias_sums_ = nullptr;
    }
    if (bias_ptr_ != nullptr) {
      ctx_->allocator->Free(weight_bias_sums_);
      weight_bias_sums_ = nullptr;
    }
  }
  MatmulQuantArg quant_params_;
  int8_t *a_r4x16_ptr_ = nullptr;
  int8_t *b_c16x4_ptr_ = nullptr;
  int *input_sums_ = nullptr;
  int *weight_bias_sums_ = nullptr;
  int *bias_ptr_ = nullptr;
  int r4_ = 0;
  int c4_ = 0;
  int d16_ = 0;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_FULLCONNECTION_INT8_H_
