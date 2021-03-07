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

#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/base/quant_dtype_cast_coder.h"
#include "coder/opcoders/serializers/serializer.h"
#include "coder/utils/type_cast.h"

using mindspore::schema::PrimitiveType_QuantDTypeCast;

namespace mindspore::lite::micro {
int QuantDTypeCastCoder::Prepare(CoderContext *const context) {
  auto *param = reinterpret_cast<QuantDTypeCastParameter *>(parameter_);
  if (input_tensor_->data_type() != static_cast<TypeId>(param->srcT) ||
      output_tensor_->data_type() != static_cast<TypeId>(param->dstT)) {
    MS_LOG(ERROR) << "param data type not supported:"
                  << " src: " << param->srcT << " dst: " << param->dstT;
    return RET_ERROR;
  }
  src_dtype = static_cast<TypeId>(param->srcT);
  dst_dtype = static_cast<TypeId>(param->dstT);
  return RET_OK;
}

int QuantDTypeCastCoder::DoCode(CoderContext *const context) {
  if (input_tensor_->quant_params().empty() && output_tensor_->quant_params().empty()) {
    MS_LOG(ERROR) << "QuantDTypeCast need quantization parameters which is not found.";
    return RET_ERROR;
  }
  auto quant_arg = (!output_tensor_->quant_params().empty() && output_tensor_->quant_params().at(0).inited)
                     ? output_tensor_->quant_params().at(0)
                     : input_tensor_->quant_params().at(0);
  int num_unit_thread = input_tensor_->ElementsNum();

  Collect(context, {"nnacl/int8/quant_dtype_cast_int8.h"}, {"quant_dtype_cast_int8.c"});
  Serializer code;
  code.precision(kPrecision);
  if (src_dtype == TypeId::kNumberTypeInt8 && dst_dtype == TypeId::kNumberTypeFloat32) {
    code.CodeFunction("DoDequantizeInt8ToFp32", input_tensor_, output_tensor_, quant_arg.scale, quant_arg.zeroPoint,
                      num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeFloat32 && dst_dtype == TypeId::kNumberTypeInt8) {
    bool from_uint8_src = false;
    if (quant_arg.dstDtype == TypeId::kNumberTypeUInt8) {
      from_uint8_src = true;
    }
    code.CodeFunction("DoQuantizeFp32ToInt8", input_tensor_, output_tensor_, quant_arg.scale, quant_arg.zeroPoint,
                      num_unit_thread, from_uint8_src);
  } else if (src_dtype == TypeId::kNumberTypeInt8 && dst_dtype == TypeId::kNumberTypeUInt8) {
    code.CodeFunction("Int8ToUInt8", input_tensor_, output_tensor_, num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeUInt8 && dst_dtype == TypeId::kNumberTypeFloat32) {
    code.CodeFunction("DoDequantizeUInt8ToFp32", input_tensor_, output_tensor_, quant_arg.scale, quant_arg.zeroPoint,
                      num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeFloat32 && dst_dtype == TypeId::kNumberTypeUInt8) {
    code.CodeFunction("DoQuantizeFp32ToUInt8", input_tensor_, output_tensor_, quant_arg.scale, quant_arg.zeroPoint,
                      num_unit_thread);
  } else if (src_dtype == TypeId::kNumberTypeUInt8 && dst_dtype == TypeId::kNumberTypeInt8) {
    code.CodeFunction("UInt8ToInt8", input_tensor_, output_tensor_, num_unit_thread);
  } else {
    MS_LOG(INFO) << "unsupported type cast, src: " << EnumNameDataType(src_dtype)
                 << ", dst: " << EnumNameDataType(dst_dtype);
    return RET_ERROR;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_QuantDTypeCast,
                   CPUOpCoderCreator<QuantDTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_QuantDTypeCast, CPUOpCoderCreator<QuantDTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeUInt8, PrimitiveType_QuantDTypeCast, CPUOpCoderCreator<QuantDTypeCastCoder>)
}  // namespace mindspore::lite::micro
