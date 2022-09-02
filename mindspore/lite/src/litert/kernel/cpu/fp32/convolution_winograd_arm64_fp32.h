/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_ARM64_FP32_CONVOLUTION_WINOGRAD_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_ARM64_FP32_CONVOLUTION_WINOGRAD_FP32_H_

#include <vector>
#include "src/litert/kernel/cpu/fp32/convolution_winograd_base_fp32.h"

namespace mindspore::kernel {
class ConvolutionWinogradARM64CPUKernel : public ConvolutionWinogradBaseCPUKernel {
 public:
  ConvolutionWinogradARM64CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                    int output_unit, float *origin_weight, float *origin_bias)
      : ConvolutionWinogradBaseCPUKernel(parameter, inputs, outputs, ctx, output_unit, origin_weight, origin_bias) {}
  ~ConvolutionWinogradARM64CPUKernel() override {}
  void InitGlobalVariable() override;
  int ConfigInputOutput() override;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_ARM64_FP32_CONVOLUTION_WINOGRAD_FP32_H_
