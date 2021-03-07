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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/common/utils.h"
#include "nnacl/int8/conv_int8.h"

namespace mindspore::kernel {
class ConvolutionInt8CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~ConvolutionInt8CPUKernel() override {
    FreeQuantParam();
    if (packed_weight_ != nullptr) {
      free(packed_weight_);
      packed_weight_ = nullptr;
    }
    if (input_sum_ != nullptr) {
      free(input_sum_);
      input_sum_ = nullptr;
    }
    if (filter_zp_ptr_ != nullptr) {
      free(filter_zp_ptr_);
      filter_zp_ptr_ = nullptr;
    }
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  void CheckSupportOptimize();
  int InitWeightBias();
  int InitTmpBuffer();

 private:
  void FreeTmpBuffer() {
    if (packed_input_ != nullptr) {
      ctx_->allocator->Free(packed_input_);
      packed_input_ = nullptr;
    }
    if (matmul_packed_input_ != nullptr) {
      ctx_->allocator->Free(matmul_packed_input_);
      matmul_packed_input_ = nullptr;
    }
    if (tmp_dst_ != nullptr) {
      ctx_->allocator->Free(tmp_dst_);
      tmp_dst_ = nullptr;
    }
    if (tmp_out_ != nullptr) {
      ctx_->allocator->Free(tmp_out_);
      tmp_out_ = nullptr;
    }
  }
  bool support_optimize_ = true;
  int8_t *packed_weight_ = nullptr;
  int8_t *packed_input_ = nullptr;
  int8_t *matmul_packed_input_ = nullptr;
  int32_t *filter_zp_ptr_ = nullptr; /* per-oc */
  int32_t *input_sum_ = nullptr;
  int32_t *tmp_dst_ = nullptr;
  int8_t *tmp_out_ = nullptr;
  MATMUL_OPT_R_FUNC matmul_func_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_INT8_H_
