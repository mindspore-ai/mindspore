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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_DELEGATE_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_DELEGATE_FP32_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"
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
      if (matmul_param_ != nullptr) {
        free(op_parameter_);
      }
      op_parameter_ = nullptr;  // op_parameter will be freed in conv_kernel
      delete conv_kernel_;
      conv_kernel_ = nullptr;
    }
    if (matmul_param_ != nullptr) {
      matmul_param_ = nullptr;
    }
  };
  int Prepare() override;
  int ReSize() override;
  bool CheckInputsValid() const override;
  int Run() override {
    conv_kernel_->set_name(name_);
    conv_kernel_->set_workspace(workspace());
    return conv_kernel_->Run();
  }

  void set_in_tensors(const std::vector<lite::Tensor *> &in_tensors) override {
    this->in_tensors_ = in_tensors;
    if (conv_kernel_ != nullptr) {
      conv_kernel_->set_in_tensors(in_tensors);
    }
  }

  void set_in_tensor(lite::Tensor *in_tensor, size_t index) override {
    MS_ASSERT(index < in_tensors_.size());
    this->in_tensors_[index] = in_tensor;
    if (conv_kernel_ != nullptr) {
      conv_kernel_->set_in_tensor(in_tensor, index);
    }
  }

  void set_out_tensors(const std::vector<lite::Tensor *> &out_tensors) override {
    this->out_tensors_ = out_tensors;
    if (conv_kernel_ != nullptr) {
      conv_kernel_->set_out_tensors(out_tensors);
    }
  }

  void set_out_tensor(lite::Tensor *out_tensor, size_t index) override {
    MS_ASSERT(index < out_tensors_.size());
    this->out_tensors_[index] = out_tensor;
    if (conv_kernel_ != nullptr) {
      conv_kernel_->set_out_tensor(out_tensor, index);
    }
  }

 protected:
  int GetWeightAndBias();
  int GetWeightData();
  int GetBiasData();

  int SetInputOutputShapeInfo();
  kernel::LiteKernel *CpuConvFp32KernelSelect();
  kernel::LiteKernel *CpuConvFp32NC4KernelSelect();
  kernel::LiteKernel *CpuConvFp32NHWCKernelSelect();
  kernel::LiteKernel *CreateConv1x1MatmulKernel();
  bool CheckAvxUseSW1x1Conv(const ConvParameter *conv_param);
  bool CheckAvxUseSWConv(const ConvParameter *conv_param);
  // If inferShape process can't complete in Init part, initialization of weight and bis will be implemented in runtime
  // via Resize() API. However,data of const tensor(weight and bias) doesn't exist anymore in runtime stage.Thus,
  // copying data of const tensor is necessary. Otherwise, just pass origin raw pointer of data.
  static float *CopyData(const lite::Tensor *tensor);
  void FreeCopiedData() {
    if (origin_weight_ != nullptr && need_free_weight_) {
      free(origin_weight_);
      origin_weight_ = nullptr;
      need_free_weight_ = false;
    }
    if (origin_bias_ != nullptr && need_free_bias_) {
      free(origin_bias_);
      origin_bias_ = nullptr;
      need_free_bias_ = false;
    }
  }
  // Train API
  int Train() override {
    (void)LiteKernel::Train();
    return conv_kernel_->Train();
  }
  void SetTrainable(bool trainable) override {
    LiteKernel::SetTrainable(trainable);
    return conv_kernel_->SetTrainable(trainable);
  }
  size_t workspace_size() override {
    (void)LiteKernel::workspace_size();
    return conv_kernel_->workspace_size();
  }

 protected:
  kernel::LiteKernel *conv_kernel_{nullptr};
  MatMulParameter *matmul_param_{nullptr};
  float *origin_weight_{nullptr};
  float *origin_bias_{nullptr};
  bool need_free_weight_{false};
  bool need_free_bias_{false};
  bool input_const_{false};
  bool weight_const_{false};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_DELEGATE_FP32_H_
