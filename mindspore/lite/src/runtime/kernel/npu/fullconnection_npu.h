/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_FULLCONNECTION_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_FULLCONNECTION_NPU_H_
#include <vector>
#include "include/graph/op/all_ops.h"
#include "src/runtime/kernel/npu/convolution_base_npu.h"
#include "nnacl/matmul_parameter.h"
namespace mindspore::kernel {
class FullconnectionNPUKernel : public ConvolutionBaseNPUKernel {
 public:
  FullconnectionNPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : ConvolutionBaseNPUKernel(parameter, inputs, outputs, ctx) {
    fc_param_ = reinterpret_cast<MatMulParameter *>(parameter);
  }
  ~FullconnectionNPUKernel() override;

  int IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                OpParameter *opParameter) override;
  int SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                   const std::vector<ge::Operator *> &npu_inputs) override;
  ge::Operator *GetNPUOp() override;

 private:
  hiai::op::Reshape *reshape_ = nullptr;
  hiai::op::MatMul *fc_ = nullptr;
  hiai::op::BiasAdd *biasadd_ = nullptr;
  hiai::op::Const *reshape_op_ = nullptr;
  MatMulParameter *fc_param_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_NPU_FULLCONNECTION_NPU_H_
