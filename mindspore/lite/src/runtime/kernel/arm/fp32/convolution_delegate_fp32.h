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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_DELEGATE_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_DELEGATE_FP32_H_

#include <vector>
#include "src/lite_kernel.h"
#include "src/runtime/kernel/arm/fp32/convolution_creator_manager.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

using mindspore::lite::InnerContext;
namespace mindspore::kernel {
class ConvolutionDelegateCPUKernel : public LiteKernel {
 public:
  ConvolutionDelegateCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~ConvolutionDelegateCPUKernel() override {
    FreeCopiedData();
    if (conv_kernel_ != nullptr) {
      op_parameter_ = nullptr;  // op_parameter will be freed in conv_kernel
      delete conv_kernel_;
      conv_kernel_ = nullptr;
    }
  };
  int Init() override;
  int ReSize() override;
  int Run() override { return conv_kernel_->Run(); }

 protected:
  int GetWeightAndBias();
  int GetWeightData();
  int GetBiasData();

  void SetInputOutputShapeInfo();
  kernel::LiteKernel *CpuConvFp32KernelSelect();

  // If inferShape process can't complete in Init part, initialization of weight and bis will be implemented in runtime
  // via Resize() API. However,data of const tensor(weight and bias) doesn't exist anymore in runtime stage.Thus,
  // copying data of const tensor is necessary. Otherwise, just pass origin raw pointer of data.
  static float *CopyData(lite::Tensor *tensor);
  void FreeCopiedData() {
    if (origin_weight_ != nullptr && need_free_weight_) {
      free(origin_weight_);
      origin_weight_ = nullptr;
    }
    if (origin_bias_ != nullptr && need_free_bias_) {
      free(origin_bias_);
      origin_bias_ = nullptr;
    }
  }

  // Train API
  int Eval() override {
    LiteKernel::Eval();
    return conv_kernel_->Eval();
  }
  int Train() override {
    LiteKernel::Train();
    return conv_kernel_->Train();
  }
  void set_trainable(bool trainable) override {
    LiteKernel::set_trainable(trainable);
    return conv_kernel_->set_trainable(trainable);
  }

 protected:
  kernel::LiteKernel *conv_kernel_{nullptr};
  float *origin_weight_{nullptr};
  float *origin_bias_{nullptr};
  bool need_free_weight_{false};
  bool need_free_bias_{false};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_DELEGATE_FP32_H_
