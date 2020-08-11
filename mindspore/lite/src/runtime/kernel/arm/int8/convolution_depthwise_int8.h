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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_DEPTHWISE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_DEPTHWISE_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/nnacl/fp32/conv_depthwise.h"

namespace mindspore::kernel {
class ConvolutionDepthwiseInt8CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionDepthwiseInt8CPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                                    const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                                    const lite::Primitive *primitive)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, primitive) {}
  ~ConvolutionDepthwiseInt8CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

  int InitWeightBias();
  int InitBuffer();
  int Execute(int task_id);

 private:
  SlidingWindowParam *sliding;
  int16_t *packed_weight_;
  int16_t *packed_input_;
  int8_t *packed_output_;
  bool need_align_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_DEPTHWISE_INT8_H_
