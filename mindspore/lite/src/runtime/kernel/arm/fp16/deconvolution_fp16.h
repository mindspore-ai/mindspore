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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_H_

#include <vector>
#include "nnacl/fp16/deconv_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"

namespace mindspore::kernel {
class DeConvolutionFp16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  DeConvolutionFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                             TypeId origin_weight_data_type, TypeId origin_bias_data_type)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, origin_weight_data_type, origin_bias_data_type) {}
  ~DeConvolutionFp16CPUKernel() override;
  int Init() override;
  int Run() override;
  int ReSize() override;

 public:
  int DoDeconv(int task_id);

 private:
  int InitRunBuf();
  void FreeRunBuf();
  int InitParam();
  int InitWeightBias();

 private:
  MatMulParameter *matmul_param_;
  int input_plane_;
  int kernel_plane_;
  int output_plane_;
  int thread_count_;
  int thread_stride_;
  float16_t *pack_input_;
  float16_t *pack_output_;
  float16_t *tmp_buffer_;
  float16_t *batch_input_;
  float16_t *batch_output_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_H_
