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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_WINOGRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_WINOGRAD_H_

#include <vector>
#include "include/errorcode.h"
#include "nnacl/fp16/common_func_fp16.h"
#include "nnacl/fp16/deconv_winograd_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"

namespace mindspore::kernel {
class DeConvWinogradFp16CPUKernel : public ConvolutionBaseFP16CPUKernel {
 public:
  DeConvWinogradFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                              TypeId origin_weight_data_type, TypeId origin_bias_data_type)
      : ConvolutionBaseFP16CPUKernel(parameter, inputs, outputs, ctx, origin_weight_data_type, origin_bias_data_type) {}
  ~DeConvWinogradFp16CPUKernel() override;
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

 private:
  DeConvParam *deconv_param_;
  std::mutex lock_;
  float16_t *nhwc_input_ = nullptr;
  float16_t *nhwc_output_ = nullptr;
  float16_t *nc4hw4_output_ = nullptr;
  float16_t *tile_input_ = nullptr;
  float16_t *tile_output_ = nullptr;
  int thread_num_hw_;
  int thread_stride_hw_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_WINOGRAD_H_
