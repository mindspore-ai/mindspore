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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DELEGATE_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DELEGATE_FP16_H_

#include <arm_neon.h>
#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

#define WEIGHT_NEED_FREE 0001
#define BIAS_NEED_FREE 0010

namespace mindspore::kernel {
class ConvolutionDelegateFP16CPUKernel : public InnerKernel {
 public:
  ConvolutionDelegateFP16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~ConvolutionDelegateFP16CPUKernel() override {
    FreeCopiedData();
    if (fp16_conv_kernel_ != nullptr) {
      op_parameter_ = nullptr;  // set op_parameter of delegate to nullptr, avoiding double free
      delete fp16_conv_kernel_;
      fp16_conv_kernel_ = nullptr;
    }
  }
  void *CopyData(const lite::Tensor *tensor);
  void FreeCopiedData();
  int Init() override;
  int ReSize() override;
  int Run() override {
    fp16_conv_kernel_->set_workspace(workspace());
    return fp16_conv_kernel_->Run();
  }
  int Train() override {
    InnerKernel::Train();
    return fp16_conv_kernel_->Train();
  }
  void SetTrainable(bool trainable) override {
    InnerKernel::SetTrainable(trainable);
    return fp16_conv_kernel_->SetTrainable(trainable);
  }
  size_t workspace_size() override {
    InnerKernel::workspace_size();
    return fp16_conv_kernel_->workspace_size();
  }

  void set_in_tensor(lite::Tensor *in_tensor, size_t index) override {
    MS_ASSERT(index < in_tensors_.size());
    this->in_tensors_[index] = in_tensor;
    if (fp16_conv_kernel_ != nullptr) {
      fp16_conv_kernel_->set_in_tensor(in_tensor, index);
    }
  }

  void set_out_tensor(lite::Tensor *out_tensor, size_t index) override {
    MS_ASSERT(index < out_tensors_.size());
    this->out_tensors_[index] = out_tensor;
    if (fp16_conv_kernel_ != nullptr) {
      fp16_conv_kernel_->set_out_tensor(out_tensor, index);
    }
  }

 private:
  kernel::InnerKernel *CpuConvFp16KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const lite::InnerContext *ctx, void *origin_weight, void *origin_bias);
  uint8_t need_free_ = 0b00;
  void *origin_weight_ = nullptr;
  void *origin_bias_ = nullptr;
  kernel::InnerKernel *fp16_conv_kernel_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_CONVOLUTION_DELEGATE_FP16_H_
