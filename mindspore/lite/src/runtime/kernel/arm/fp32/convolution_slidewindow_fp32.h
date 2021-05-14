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
#ifdef ENABLE_AVX

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/op_base.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"

namespace mindspore::kernel {
class ConvolutionSWCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionSWCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                         float *origin_weight, float *origin_bias)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx),
        ori_weight_data_(origin_weight),
        ori_bias_data_(origin_bias) {}

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
  void PackNHWCTo1HWCNX(int kernel_h, int kernel_w, int output_channel, int oc_block_num, int input_channel,
                        float *tmp_weight);

 private:
  void FreeTmpBuffer() {
    if (output_data_ != nullptr && oc_res_ != 0) {
      ctx_->allocator->Free(output_data_);
      output_data_ = nullptr;
    }
  }
  int oc_tile_ = C8NUM;  // oc tile is C8NUM in avx
  int oc_res_ = 0;
  float *ori_input_data_ = nullptr;
  float *ori_weight_data_ = nullptr;
  float *ori_bias_data_ = nullptr;
  float *packed_weight_ = nullptr;
  float *output_data_ = nullptr;
  SlidingWindowParam *slidingWindow_param_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // ENABLE_AVX
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_SLIDEWINDOW_H_
