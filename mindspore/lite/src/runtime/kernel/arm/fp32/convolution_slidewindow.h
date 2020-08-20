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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_SLIDEWINDOW_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_SLIDEWINDOW_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/nnacl/op_base.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/nnacl/fp32/conv.h"
#include "src/runtime/kernel/arm/nnacl/fp32/conv_depthwise.h"

namespace mindspore::kernel {
class ConvolutionSWCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionSWCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                         const std::vector<lite::tensor::Tensor *> &outputs, const lite::Context *ctx,
                         const mindspore::lite::PrimitiveC *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}

  ~ConvolutionSWCPUKernel() override {
    if (packed_weight_ != nullptr) {
      free(packed_weight_);
      packed_weight_ = nullptr;
    }
    if (slidingWindow_param_ != nullptr) {
      delete slidingWindow_param_;
      slidingWindow_param_ = nullptr;
    }
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int InitTmpBuffer();
  void ConfigInputOutput();

 private:
  void FreeTmpBuffer() {
    if (tmp_output_block_ != nullptr) {
      ctx_->allocator->Free(tmp_output_block_);
      tmp_output_block_ = nullptr;
    }
  }
  float *packed_weight_ = nullptr;
  float *tmp_output_block_ = nullptr;
  SlidingWindowParam *slidingWindow_param_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_SLIDEWINDOW_H_
