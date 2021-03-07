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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_ACTIVATION_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_ACTIVATION_NPU_H_

#include <vector>
#include "include/graph/op/all_ops.h"
#include "include/graph/compatible/all_ops.h"
#include "src/runtime/kernel/npu/npu_kernel.h"
#include "nnacl/fp32/activation_fp32.h"

namespace mindspore::kernel {
class ActivationNPUKernel : public NPUKernel {
 public:
  ActivationNPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : NPUKernel(parameter, inputs, outputs, ctx) {
    act_param_ = reinterpret_cast<ActivationParameter *>(parameter);
  }
  ~ActivationNPUKernel() override;

  int IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                OpParameter *opParameter) override;
  int SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                   const std::vector<ge::Operator *> &npu_inputs) override;
  ge::Operator *GetNPUOp() override;

 private:
  hiai::op::Activation *act_ = nullptr;
  ActivationParameter *act_param_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_ACTIVATION_NPU_H_
