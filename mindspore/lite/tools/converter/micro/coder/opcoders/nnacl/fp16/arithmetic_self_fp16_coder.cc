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

#include "coder/opcoders/nnacl/fp16/arithmetic_self_fp16_coder.h"
#include <map>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

namespace mindspore::lite::micro::nnacl {
int ArithmeticSelfFP16Coder::Prepare(CoderContext *const context) {
  if (parameter_ == nullptr) {
    return RET_ERROR;
  }
  if (input_tensor_->data_type() != kNumberTypeFloat16 || output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  std::map<int, std::function<void()>> type_setters = {
    {PrimitiveType_Abs, [this]() { arithmetic_self_run_ = "ElementAbsFp16"; }},
    {PrimitiveType_Cos, [this]() { arithmetic_self_run_ = "ElementCosFp16"; }},
    {PrimitiveType_Log, [this]() { arithmetic_self_run_ = "ElementLogFp16"; }},
    {PrimitiveType_Square, [this]() { arithmetic_self_run_ = "ElementSquareFp16"; }},
    {PrimitiveType_Sqrt, [this]() { arithmetic_self_run_ = "ElementSqrtFp16"; }},
    {PrimitiveType_Rsqrt, [this]() { arithmetic_self_run_ = "ElementRsqrtFp16"; }},
    {PrimitiveType_Sin, [this]() { arithmetic_self_run_ = "ElementSinFp16"; }},
    {PrimitiveType_LogicalNot, [this]() { arithmetic_self_run_ = "ElementLogicalNotFp16"; }},
    {PrimitiveType_Floor, [this]() { arithmetic_self_run_ = "ElementFloorFp16"; }},
    {PrimitiveType_Ceil, [this]() { arithmetic_self_run_ = "ElementCeilFp16"; }},
    {PrimitiveType_Round, [this]() { arithmetic_self_run_ = "ElementRoundFp16"; }},
    {PrimitiveType_Neg, [this]() { arithmetic_self_run_ = "ElementNegativeFp16"; }},
    {PrimitiveType_Erf, [this]() { arithmetic_self_run_ = "ElementErfFp16"; }},
  };
  auto iter = type_setters.find(parameter_->type_);
  if (iter != type_setters.end()) {
    iter->second();
  } else {
    MS_LOG(ERROR) << "Error Operator type " << parameter_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfFP16Coder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(!arithmetic_self_run_.empty(), "arithmetic_run function is nullptr!");

  Collect(context,
          {
            "nnacl/fp16/arithmetic_self_fp16.h",
          },
          {
            "arithmetic_self_fp16.c",
          });
  NNaclFp32Serializer code;
  code.CodeFunction(arithmetic_self_run_, input_tensor_, output_tensor_, input_tensor_->ElementsNum());

  MS_LOG(DEBUG) << "ArithmeticSelfFP16Coder has been called";
  context->AppendCode(code.str());

  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Abs, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Abs, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Cos, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Cos, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Log, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Log, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Square, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Square, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Sqrt, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Sqrt, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Rsqrt, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Rsqrt, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Sin, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Sin, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LogicalNot, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LogicalNot, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Floor, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Floor, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Ceil, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Ceil, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Round, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Round, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Neg, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Neg, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Erf, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Erf, CPUOpCoderCreator<ArithmeticSelfFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
