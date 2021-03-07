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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/op_base.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"

namespace mindspore::kernel {
class ConvolutionCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx, float *origin_weight,
                       float *origin_bias)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx),
        origin_weight_(origin_weight),
        origin_bias_(origin_bias) {}
  ~ConvolutionCPUKernel() override {
    if (packed_weight_ != nullptr) {
      free(packed_weight_);
      packed_weight_ = nullptr;
    }
  }

  int Init() override;
  virtual int InitWeightBias();
  int InitTmpBuffer();
  int ReSize() override;
  int Run() override;
  virtual int RunImpl(int task_id);

  int Eval() override;

 protected:
  void PackWeight();
  void FreeTmpBuffer() {
    if (packed_input_ != nullptr) {
      ctx_->allocator->Free(packed_input_);
      packed_input_ = nullptr;
    }
    if (col_major_input_ != nullptr) {
      ctx_->allocator->Free(col_major_input_);
      col_major_input_ = nullptr;
    }
  }

 protected:
  float *origin_weight_;  // do not free
  float *origin_bias_;    // do not free
  float *packed_weight_ = nullptr;
  float *packed_input_ = nullptr;
  float *col_major_input_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_H_
