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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DECONVOLUTION_WINOGRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DECONVOLUTION_WINOGRAD_H_

#include <float.h>
#include <vector>
#include "src/lite_kernel.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/deconv_winograd_fp32.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"

namespace mindspore::kernel {
class DeConvolutionWinogradCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  DeConvolutionWinogradCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~DeConvolutionWinogradCPUKernel() override;
  int Init() override;
  int Run() override;
  int ReSize() override;

 public:
  int DoDeconv(int task_id);
  int DeDeconvPost(int task_id);

 private:
  int InitComputeParam();
  int InitDataParam();
  int InitParameter();
  void FreeDeconvParam();
  void FreeResizeBuf();
  int InitRunBuf();
  void FreeRunBuf();

 private:
  DeConvParam *deconv_param_ = nullptr;
  float *nhwc_input_ = nullptr;
  float *nhwc_output_ = nullptr;
  float *nc4hw4_output_ = nullptr;
  float *tile_input_ = nullptr;
  float *tile_output_ = nullptr;
  std::mutex lock_;
  int thread_num_hw_ = 0;
  int thread_stride_hw_ = 0;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_DECONVOLUTION_WINOGRAD_H_
