/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp16/activation_fp16_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::lite::micro::nnacl {
int ActivationFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(INFO) << "Input tensor data type is invalid";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

int ActivationFP16Coder::DoCode(CoderContext *const context) {
  // attribute
  auto *activation_parameter = reinterpret_cast<ActivationParameter *>(parameter_);
  int count = input_tensor_->ElementsNum();
  Collect(context,
          {
            "nnacl/fp16/activation_fp16.h",
          },
          {
            "activation_fp16.c",
          });
  NNaclFp32Serializer code;

  switch (activation_parameter->type_) {
    case schema::ActivationType_RELU:
      code.CodeFunction("ReluFp16", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_RELU6:
      code.CodeFunction("Relu6Fp16", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_LEAKY_RELU:
      code.CodeFunction("LReluFp16", input_tensor_, count, output_tensor_, activation_parameter->alpha_);
      break;
    case schema::ActivationType_SIGMOID:
      code.CodeFunction("SigmoidFp16", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_TANH:
      code.CodeFunction("TanhFp16", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_HSWISH:
      code.CodeFunction("HSwishFp16", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_SWISH:
      code.CodeFunction("SwishFp16", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_HSIGMOID:
      code.CodeFunction("HSigmoidFp16", input_tensor_, count, output_tensor_);
      break;
    case schema::ActivationType_ELU:
      code.CodeFunction("EluFp16", input_tensor_, count, output_tensor_, activation_parameter->alpha_);
      break;
    default:
      MS_LOG(ERROR) << "Activation type error";
      return RET_ERROR;
  }
  MS_LOG(DEBUG) << "ActivationFP16Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Activation, CPUOpCoderCreator<ActivationFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
