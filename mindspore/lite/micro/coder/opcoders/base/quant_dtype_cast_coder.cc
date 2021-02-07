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

#include <string>
#include "micro/coder/opcoders/op_coder.h"
#include "micro/coder/opcoders/file_collector.h"
#include "micro/coder/opcoders/base/quant_dtype_cast_coder.h"
#include "micro/coder/opcoders/serializers/serializer.h"

using mindspore::schema::PrimitiveType_QuantDTypeCast;

namespace mindspore::lite::micro {

int QuantDTypeCastCoder::Prepare(CoderContext *const context) {
  this->cast_param_ = reinterpret_cast<QuantDTypeCastParameter *>(parameter_);

  if (cast_param_->srcT == kNumberTypeFloat32 && cast_param_->dstT == kNumberTypeInt8) {
    if (input_tensor_->data_type() != kNumberTypeFloat32 || output_tensor_->data_type() != kNumberTypeInt8) {
      MS_LOG(ERROR) << "cast_param_ data type and tensor data type do not match.";
      return RET_ERROR;
    }
    inverse_ = false;
  } else if (cast_param_->srcT == kNumberTypeInt8 && cast_param_->dstT == kNumberTypeFloat32) {
    if (input_tensor_->data_type() != kNumberTypeInt8 || output_tensor_->data_type() != kNumberTypeFloat32) {
      MS_LOG(ERROR) << "cast_param_ data type and tensor data type do not match.";
      return RET_ERROR;
    }
    inverse_ = true;
  } else {
    MS_LOG(ERROR) << "cast_param_ data type not supported:"
                  << " src: " << cast_param_->srcT << " dst: " << cast_param_->dstT;
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

int QuantDTypeCastCoder::DoCode(CoderContext *const context) {
  // get quant params
  QuantArg in_quant_arg = input_tensor_->quant_params().at(0);

  // single thread for now
  int num_unit_thread = input_tensor_->ElementsNum();

  // generate code .h .c
  Collect(context, {"nnacl/int8/quant_dtype_cast_int8.h"}, {"quant_dtype_cast_int8.c"});

  Serializer code;
  code.precision(kPrecision);
  std::string function = inverse_ ? "DoDequantizeInt8ToFp32" : "DoQuantizeFp32ToInt8";
  code.CodeFunction(function, input_tensor_, output_tensor_, in_quant_arg.scale, in_quant_arg.zeroPoint,
                    num_unit_thread);

  context->AppendCode(code.str());

  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_QuantDTypeCast,
                   CPUOpCoderCreator<QuantDTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_QuantDTypeCast, CPUOpCoderCreator<QuantDTypeCastCoder>)
}  // namespace mindspore::lite::micro
