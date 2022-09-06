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

#include "coder/opcoders/nnacl/int8/arithmetic_self_int8_coder.h"
#include <algorithm>
#include <limits>
#include "include/errorcode.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Abs;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Reciprocal;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore::lite::micro::nnacl {
int ArithmeticSelfInt8Coder::Prepare(CoderContext *context) {
  CHECK_LESS_RETURN(input_tensors_.size(), kInputIndex + 1);
  CHECK_LESS_RETURN(output_tensors_.size(), kOutputIndex + 1);
  auto *input_tensor = input_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  auto in_quant_args = input_tensor->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);

  param_.quant_arg_.in_args_.scale_ = in_quant_args.front().scale;
  param_.quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint * (-1);

  auto *out_tensor = output_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  param_.quant_arg_.out_args_.scale_ = out_quant_args.front().scale;
  param_.quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  param_.quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param_.quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();

  if (param_.op_parameter_.type_ == PrimitiveType_Square) {
    const double real_multiplier =
      (param_.quant_arg_.in_args_.scale_ * param_.quant_arg_.in_args_.scale_) / param_.quant_arg_.out_args_.scale_;

    int right_shift = 0;
    QuantizeMultiplierSmallerThanOne(real_multiplier, &param_.quant_arg_.output_multiplier_, &right_shift);

    param_.quant_arg_.shift_left_ = right_shift < 0 ? -right_shift : 0;
    param_.quant_arg_.shift_right_ = right_shift > 0 ? right_shift : 0;
  }

  switch (parameter_->type_) {
    case PrimitiveType_Round:
      arithmeticSelf_run_ = "Int8ElementRound";
      break;
    case PrimitiveType_Floor:
      arithmeticSelf_run_ = "Int8ElementFloor";
      break;
    case PrimitiveType_Ceil:
      arithmeticSelf_run_ = "Int8ElementCeil";
      break;
    case PrimitiveType_Abs:
      arithmeticSelf_run_ = "Int8ElementAbs";
      break;
    case PrimitiveType_Sin:
      arithmeticSelf_run_ = "Int8ElementSin";
      break;
    case PrimitiveType_Cos:
      arithmeticSelf_run_ = "Int8ElementCos";
      break;
    case PrimitiveType_Log:
      arithmeticSelf_run_ = "Int8ElementLog";
      break;
    case PrimitiveType_Sqrt:
      arithmeticSelf_run_ = "Int8ElementSqrt";
      break;
    case PrimitiveType_Rsqrt:
      arithmeticSelf_run_ = "Int8ElementRsqrt";
      break;
    case PrimitiveType_Square:
      arithmeticSelf_run_ = "Int8ElementSquare";
      break;
    case PrimitiveType_LogicalNot:
      arithmeticSelf_run_ = "Int8ElementLogicalNot";
      break;
    case PrimitiveType_Reciprocal:
      arithmeticSelf_run_ = "Int8ElementReciprocal";
      break;
    default:
      MS_LOG(ERROR) << "Unknown op type " << parameter_->type_;
      return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/int8/arithmetic_self_int8.h",
          },
          {
            "arithmetic_self_int8.c",
          });
  NNaclInt8Serializer code;
  code.CodeStruct("param", param_.quant_arg_);
  code.CodeFunction(arithmeticSelf_run_, input_tensor_, output_tensor_, input_tensor_->ElementsNum(), "param");

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Round, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Floor, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Ceil, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Abs, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Sin, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Cos, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Log, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Sqrt, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Rsqrt, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Square, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_LogicalNot, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Reciprocal, CPUOpCoderCreator<ArithmeticSelfInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
