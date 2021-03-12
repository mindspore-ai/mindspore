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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CONVOLUTION_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CONVOLUTION_BASE_H_

#include <unistd.h>
#include <vector>
#include <string>
#include <limits>
#ifdef ENABLE_ARM
#include <arm_neon.h>
#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#endif
#endif
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class ConvolutionBaseCPUKernel : public LiteKernel {
 public:
  ConvolutionBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), ctx_(ctx), thread_count_(ctx->thread_num_) {
    op_parameter_->thread_num_ = ctx->thread_num_;
    conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter_);
  }
  ~ConvolutionBaseCPUKernel() override;

  int Init() override;
  int ReSize() override { return 0; }
  int Run() override { return 0; }
  int SetIfPerChannel();
  int MallocQuantParam();
  int SetQuantParam();
  int SetInputTensorQuantParam();
  int SetFilterTensorQuantParam();
  int SetOutputTensorQuantParam();
  int SetQuantMultiplier();
  void SetRoundingAndMultipilerMode();
  int CheckResizeValid();
  void FreeQuantParam();

 protected:
  void *bias_data_ = nullptr;
  const InnerContext *ctx_ = nullptr;
  ConvParameter *conv_param_ = nullptr;
  ConvQuantArg *conv_quant_arg_ = nullptr;
  int tile_num_ = 0;
  int thread_count_ = 1;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CONVOLUTION_BASE_H_
