/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_CONVOLUTION_BASE_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_CONVOLUTION_BASE_NPU_H_

#include <vector>
#include <memory>
#include "src/runtime/kernel/npu/npu_kernel.h"
#include "include/graph/op/all_ops.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::kernel {
class ConvolutionBaseNPUKernel : public NPUKernel {
 public:
  ConvolutionBaseNPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                           const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : NPUKernel(parameter, inputs, outputs, ctx) {}
  ~ConvolutionBaseNPUKernel() override;

 protected:
  int InitWeightConst(const std::vector<lite::Tensor *> &inputs);
  int InitBiasConst(const std::vector<lite::Tensor *> &inputs);
  int SetActivation(const ge::Operator *input, ActType act_type);
  hiai::op::Activation *act_ = nullptr;
  hiai::op::Const *weight_ = nullptr;
  hiai::op::Const *bias_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_CONVOLUTION_BASE_NPU_H_
