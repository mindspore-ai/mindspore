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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DEPTHWISE_SW_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DEPTHWISE_SW_FP16_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"
#include "nnacl/fp16/conv_depthwise_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
void ConvDwC8Fp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                  const float16_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                  int task_id);
#ifdef __cplusplus
}
#endif

namespace mindspore::kernel {
class ConvolutionDepthwiseSWFp16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  ConvolutionDepthwiseSWFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx,
                                      void *origin_weight, void *origin_bias, TypeId origin_weight_data_type,
                                      TypeId origin_bias_data_type)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, origin_weight_data_type, origin_bias_data_type),
        origin_weight_(origin_weight),
        origin_bias_(origin_bias) {}
  ~ConvolutionDepthwiseSWFp16CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

  int InitPackedInputOutput();
  int InitWeightBias();
  int Execute(int task_id);

 private:
  void FreePackedInputOutput();
  void *origin_weight_;  // do not free
  void *origin_bias_;    // do not free
  SlidingWindowParam *sliding_ = nullptr;
  float16_t *packed_weight_ = nullptr;
  float16_t *packed_input_ = nullptr;
  float16_t *packed_output_ = nullptr;
  bool need_align_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DEPTHWISE_SW_FP16_H_
