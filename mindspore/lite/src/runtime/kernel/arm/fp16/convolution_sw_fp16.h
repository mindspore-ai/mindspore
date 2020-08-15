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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_SW_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_SW_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"

namespace mindspore::kernel {
class ConvolutionSWFP16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  ConvolutionSWFP16CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                             const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                             const lite::Primitive *primitive)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~ConvolutionSWFP16CPUKernel() override {
    if (fp16_input_ != nullptr) {
      free(fp16_input_);
    }
    if (fp16_weight_ != nullptr) {
      free(fp16_weight_);
    }
    if (fp16_out_ != nullptr) {
      free(fp16_out_);
    }
    if (packed_weight_ != nullptr) {
      free(packed_weight_);
    }
    if (tmp_output_block_ != nullptr) {
      free(tmp_output_block_);
    }
    delete slidingWindow_param_;
  }

  int Init() override;
  int ReSize() override;
  int Run() override;
  int RunImpl(int task_id);
  int InitWeightBias();
  int InitTmpBuffer();
  void ConfigInputOutput();
  int ProcessFilter();

 private:
  float16_t *packed_weight_;
  float16_t *tmp_output_block_;
  SlidingWindowParam *slidingWindow_param_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_SW_FP16_H_
