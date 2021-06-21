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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_BASE_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_BASE_NPU_H_

#include <vector>
#include <memory>
#include <string>
#include "include/graph/op/all_ops.h"
#include "src/delegate/npu/op/npu_op.h"
namespace mindspore {
class ConvolutionBaseNPUOp : public NPUOp {
 public:
  ConvolutionBaseNPUOp(const schema::Primitive *primitive, const std::vector<tensor::MSTensor *> &in_tensors,
                       const std::vector<tensor::MSTensor *> &out_tensors, std::string name)
      : NPUOp(primitive, in_tensors, out_tensors, name) {}

  ~ConvolutionBaseNPUOp() override;

 protected:
  int InitWeightConst(const std::vector<tensor::MSTensor *> &inputs);
  int InitBiasConst(const std::vector<tensor::MSTensor *> &inputs);
  int SetActivation(const ge::Operator *input, schema::ActivationType act_type);
  hiai::op::Activation *act_ = nullptr;
  hiai::op::Const *weight_ = nullptr;
  hiai::op::Const *bias_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_BASE_NPU_H_
