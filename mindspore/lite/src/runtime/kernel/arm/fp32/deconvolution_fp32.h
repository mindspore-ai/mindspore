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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DECONVOLUTION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DECONVOLUTION_H_

#include <float.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/fp32/deconv_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

namespace mindspore::kernel {
class DeConvolutionCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  DeConvolutionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~DeConvolutionCPUKernel() override;
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
  MatMulParameter *matmul_param_ = nullptr;
  int input_plane_ = 0;
  int kernel_plane_ = 0;
  int output_plane_ = 0;
  int thread_count_ = 1;
  int thread_stride_ = 0;
  float *weight_ptr_ = nullptr;
  float *pack_input_ = nullptr;
  float *pack_output_ = nullptr;
  float *tmp_buffer_ = nullptr;
  float *input_ptr_ = nullptr;
  float *output_ptr_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DECONVOLUTION_H_
