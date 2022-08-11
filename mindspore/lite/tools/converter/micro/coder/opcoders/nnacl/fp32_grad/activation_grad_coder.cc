/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/fp32_grad/activation_grad_coder.h"
#include "nnacl/fp32_grad/activation_grad_fp32.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_ActivationGrad;

namespace mindspore::lite::micro::nnacl {
int ActivationGradCoder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(input_tensors_.size() == DIMENSION_2D, "inputs size is not equal to two");
  Tensor *input0 = input_tensors_.at(0);
  Tensor *input1 = input_tensors_.at(1);
  // attribute
  auto *activation_parameter = reinterpret_cast<ActivationParameter *>(parameter_);
  int count = input_tensor_->ElementsNum();
  Collect(context,
          {
            "nnacl/fp32_grad/activation_grad_fp32.h",
          },
          {
            "activation_grad_fp32.c",
          });
  NNaclFp32Serializer code;

  switch (activation_parameter->type_) {
    case schema::ActivationType_RELU:
      code.CodeFunction("ReluGrad", input0, input1, count, output_tensor_);
      break;
    case schema::ActivationType_ELU:
      code.CodeFunction("EluGrad", input0, input1, count, output_tensor_, activation_parameter->alpha_);
      break;
    default:
      MS_LOG(ERROR) << "Activation type error";
      return RET_ERROR;
  }
  MS_LOG(DEBUG) << "ActivationGradCode has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ActivationGrad,
                   CPUOpCoderCreator<ActivationGradCoder>)
}  // namespace mindspore::lite::micro::nnacl
