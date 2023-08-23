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
#include "coder/opcoders/nnacl/fp16/activation_dynamic_fp16_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/coder_utils.h"
#include "tools/common/string_util.h"

using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::lite::micro::nnacl {
int ActivationDynamicFP16Coder::Prepare(CoderContext *const context) {
  MS_CHECK_TRUE_MSG(input_tensor_->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Input tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(output_tensor_->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Output tensor data type is invalid.");
  return RET_OK;
}

int ActivationDynamicFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/fp16/activation_fp16.h",
          },
          {
            "activation_fp16.c",
          });
  NNaclFp32Serializer code;
  // attribute
  auto *activation_parameter = reinterpret_cast<ActivationParameter *>(parameter_);
  MS_CHECK_PTR(activation_parameter);
  auto in_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  count_ = AccumulateShape(in_shape);
  input_data_ = dynamic_mem_manager_->GetVarTensorAddr(input_tensor_);
  MS_CHECK_TRUE_MSG(!input_data_.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  output_data_ = dynamic_mem_manager_->GetVarTensorAddr(output_tensor_);
  MS_CHECK_TRUE_MSG(!output_data_.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  input_data_ = "(float16_t *)(" + input_data_ + ")";
  output_data_ = "(float16_t *)(" + output_data_ + ")";

  switch (activation_parameter->type_) {
    case schema::ActivationType_RELU:
      code.CodeFunction("ReluFp16", input_data_, output_data_, count_);
      break;
    case schema::ActivationType_RELU6:
      code.CodeFunction("Relu6Fp16", input_data_, output_data_, count_);
      break;
    case schema::ActivationType_LEAKY_RELU:
      code.CodeFunction("LReluFp16", input_data_, output_data_, count_, activation_parameter->alpha_);
      break;
    case schema::ActivationType_SIGMOID:
      code.CodeFunction("SigmoidFp16", input_data_, output_data_, count_);
      break;
    case schema::ActivationType_TANH:
      code.CodeFunction("TanhFp16", input_data_, output_data_, count_);
      break;
    case schema::ActivationType_HSWISH:
      code.CodeFunction("HSwishFp16", input_data_, output_data_, count_);
      break;
    case schema::ActivationType_SWISH:
      code.CodeFunction("SwishFp16", input_data_, output_data_, count_);
      break;
    case schema::ActivationType_HSIGMOID:
      code.CodeFunction("HSigmoidFp16", input_data_, output_data_, count_);
      break;
    case schema::ActivationType_ELU:
      code.CodeFunction("EluFp16", input_data_, output_data_, count_, activation_parameter->alpha_);
      break;
    default:
      MS_LOG(ERROR) << "Activation type error";
      return RET_ERROR;
  }
  MS_LOG(DEBUG) << "ActivationFP16Code has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Activation,
                           CPUOpCoderCreator<ActivationDynamicFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
