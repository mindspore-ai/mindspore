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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DECONVOLUTION_DEPTHWISE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DECONVOLUTION_DEPTHWISE_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"

namespace mindspore::kernel {
class DeconvolutionDepthwiseInt8CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  DeconvolutionDepthwiseInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~DeconvolutionDepthwiseInt8CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

  int InitSlideParam();
  int InitWeightBias();
  int InitBuffer();
  int Execute(int task_id);

 private:
  SlidingWindowParam *sliding = nullptr;
  int16_t *packed_weight_ = nullptr;
  int16_t *packed_input_ = nullptr;
  int8_t *packed_output_ = nullptr;
  int32_t *output_buffer_ = nullptr;
  bool need_align_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_DECONVOLUTION_DEPTHWISE_INT8_H_
