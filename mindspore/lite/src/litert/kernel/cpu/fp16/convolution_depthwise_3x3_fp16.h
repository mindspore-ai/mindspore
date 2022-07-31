/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONVOLUTION_DEPTHWISE_3X3_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONVOLUTION_DEPTHWISE_3X3_FP16_H_

#ifdef ENABLE_ARM
#include <vector>
#include "src/litert/lite_kernel.h"
#include "src/litert/kernel/cpu/base/convolution_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"

namespace mindspore::kernel {
class ConvolutionDepthwise3x3Fp16CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionDepthwise3x3Fp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, inputs.at(kWeightIndex)->data(),
                                 inputs.size() == kInputSize2 ? inputs.at(kBiasIndex)->data() : nullptr) {}
  ~ConvolutionDepthwise3x3Fp16CPUKernel() override {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;

  int DoExecute(int task_id);

 private:
  void PackWeight() override;
  int MallocWeightBiasData() override;
  float16_t *input_ptr_ = nullptr;
  float16_t *output_ptr_ = nullptr;
  float16_t *buffer_ = nullptr;
};
}  // namespace mindspore::kernel
#endif
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP16_CONVOLUTION_DEPTHWISE_3X3_FP16_H_
