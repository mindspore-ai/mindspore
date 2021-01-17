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
#include "src/ops/conv2d.h"
#include "src/lite_kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

using mindspore::lite::InnerContext;
namespace mindspore::kernel {
class ConvolutionDelegateCPUKernel : public LiteKernel {
 public:
  ConvolutionDelegateCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                               const mindspore::lite::PrimitiveC *primitive)
      : LiteKernel(parameter, inputs, outputs, ctx, primitive) {}
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
  int GetWeightAndBias();
  int GetWeightData();
  int GetBiasData();
  static float *CopyData(lite::Tensor *tensor);
  void FreeCopiedData();

  int Eval() override {
    LiteKernel::Eval();
    return conv_kernel_->Eval();
  }
  int Train() override {
    LiteKernel::Train();
    return conv_kernel_->Train();
  }

 protected:
  bool need_free_weight_ = false;
  bool need_free_bias_ = false;
  kernel::LiteKernel *conv_kernel_ = nullptr;
  float *origin_weight_ = nullptr;
  float *origin_bias_ = nullptr;
};

void SetInputOutputShapeInfo(ConvParameter *conv_param, const lite::Tensor *input, const lite::Tensor *output,
                             const InnerContext *ctx);

void FreeMemory(const std::vector<kernel::LiteKernel *> &group_convs, const std::vector<lite::Tensor *> &new_inputs,
                const std::vector<lite::Tensor *> &new_outputs);

ConvParameter *CreateNewConvParameter(ConvParameter *parameter);

lite::Tensor *CreateInputTensor(TypeId data_type, const std::vector<int> &in_shape, bool infered_flag);

lite::Tensor *CreateOutputTensor(const std::vector<int> &out_shape, const std::vector<lite::Tensor *> &outputs,
                                 bool infered_flag, int index);

kernel::LiteKernel *CpuConvFp32KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive,
                                            float *origin_weight, float *origin_bias);
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CONVOLUTION_DELEGATE_FP32_H_
