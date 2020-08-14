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
#include <android/log.h>
#endif
#include "src/lite_kernel.h"
#include "include/context.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

using mindspore::lite::Context;
using mindspore::schema::PadMode;
using mindspore::schema::QuantType;
static constexpr int kPerTensor = 1;

namespace mindspore::kernel {
class ConvolutionBaseCPUKernel : public LiteKernel {
 public:
  ConvolutionBaseCPUKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                           const std::vector<lite::tensor::Tensor *> &outputs, const Context *ctx,
                           const lite::Primitive *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive), ctx_(ctx), thread_count_(ctx->thread_num_) {
    op_parameter_->thread_num_ = ctx->thread_num_;
    conv_param_ = reinterpret_cast<ConvParameter *>(op_parameter_);
  }
  ~ConvolutionBaseCPUKernel() override;

  int Init() override;
  int ReSize() override { return 0; }
  int Run() override { return 0; }
  virtual int CheckLayout(lite::tensor::Tensor *input_tensor);
  int SetIfAsymmetric();
  int SetIfPerChannel();
  int MallocQuantParam();
  int SetQuantParam();
  int SetInputTensorQuantParam();
  int SetFilterTensorQuantParam();
  int SetOutputTensorQuantParam();
  int SetQuantMultiplier();
  void FreeQuantParam();

 protected:
  int thread_count_;
  int tile_num_;
  void *bias_data_ = nullptr;
  void *nhwc4_input_ = nullptr;
  const Context *ctx_;
  ConvParameter *conv_param_;
  ConvQuantArg *conv_quant_arg_;
  LayoutConvertor convert_func_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_CONVOLUTION_BASE_H_
