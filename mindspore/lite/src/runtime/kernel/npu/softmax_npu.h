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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_SOFTMAX_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_SOFTMAX_NPU_H_
#include <vector>
#include "src/runtime/kernel/npu/npu_kernel.h"
#include "nnacl/softmax_parameter.h"
#include "include/graph/op/nn_defs.h"
namespace mindspore::kernel {
class SoftmaxNPUKernel : public NPUKernel {
 public:
  SoftmaxNPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                   const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : NPUKernel(parameter, inputs, outputs, ctx) {
    softmax_parameter_ = reinterpret_cast<SoftmaxParameter *>(parameter);
  }
  ~SoftmaxNPUKernel() override;

  int IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                OpParameter *opParameter) override;
  int SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                   const std::vector<ge::Operator *> &npu_inputs) override;
  ge::Operator *GetNPUOp() override;

 private:
  hiai::op::Softmax *op_ = nullptr;
  SoftmaxParameter *softmax_parameter_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_SOFTMAX_NPU_H_
