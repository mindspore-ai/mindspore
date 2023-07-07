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

#include <string>
#include <map>
#include <utility>
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/base/dtype_cast_coder.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Cast;
namespace mindspore::lite::micro {
namespace {
std::map<TypeId, std::string> cast_to_fp32_func = {{kNumberTypeUInt8, "Uint8ToFloat32"},
                                                   {kNumberTypeInt32, "Int32ToFloat32"},
                                                   {kNumberTypeInt64, "Int64ToFloat32"},
                                                   {kNumberTypeFloat16, "Fp16ToFloat32"},
                                                   {kNumberTypeBool, "BoolToFloat32"}};
std::map<TypeId, std::string> cast_to_fp16_func = {{kNumberTypeUInt8, "Uint8ToFp16"},
                                                   {kNumberTypeInt32, "Int32ToFp16"},
                                                   {kNumberTypeInt64, "Int64ToFp16"},
                                                   {kNumberTypeFloat32, "Float32ToFp16"},
                                                   {kNumberTypeBool, "BoolToFp16"}};
std::map<std::pair<TypeId, TypeId>, std::string> cast_to_other_func = {
  {std::make_pair(kNumberTypeFloat32, kNumberTypeInt64), "Float32ToInt64"},
  {std::make_pair(kNumberTypeFloat32, kNumberTypeInt32), "Float32ToInt32"},
  {std::make_pair(kNumberTypeInt32, kNumberTypeInt64), "Int32ToInt64"},
  {std::make_pair(kNumberTypeInt64, kNumberTypeInt32), "Int64ToInt32"},
  {std::make_pair(kNumberTypeFloat32, kNumberTypeInt16), "Float32ToInt16"},
  {std::make_pair(kNumberTypeBool, kNumberTypeInt32), "BoolToInt32"},
  {std::make_pair(kNumberTypeFloat32, kNumberTypeBool), "Float32ToBool"},
  {std::make_pair(kNumberTypeFloat32, kNumberTypeUInt8), "Float32ToUint8"},
};
}  // namespace
int DTypeCastCoder::Prepare(CoderContext *const context) {
  data_num_ = input_tensor_->ElementsNum();
  if (data_num_ == 0) {
    return RET_OK;
  }
  int thread_num = MSMIN(thread_num_, static_cast<int>(data_num_));
  MS_CHECK_TRUE(thread_num > 0, "thread_num <= 0");
  stride_ = UP_DIV(data_num_, thread_num);
  return RET_OK;
}

int DTypeCastCoder::CastToFloat32(CoderContext *const context, TypeId input_data_type, TypeId output_data_type,
                                  const int data_num) {
  Serializer code;
  if (cast_to_fp32_func.find(input_data_type) == cast_to_fp32_func.end()) {
    MS_LOG(ERROR) << "Unsupported datatype from " << input_data_type << " cast to " << output_data_type;
    return RET_ERROR;
  } else {
    code.CodeFunction(cast_to_fp32_func[input_data_type], input_tensor_, output_tensor_, data_num);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

int DTypeCastCoder::CastToFloat16(CoderContext *const context, TypeId input_data_type, TypeId output_data_type,
                                  const int data_num) {
  Serializer code;
  code << "#ifdef ENABLE_FP16\n";
  if (cast_to_fp16_func.find(input_data_type) == cast_to_fp16_func.end()) {
    MS_LOG(ERROR) << "Unsupported datatype from " << input_data_type << " cast to " << output_data_type;
    return RET_ERROR;
  } else {
    code.CodeFunction(cast_to_fp16_func[input_data_type], input_tensor_, output_tensor_, data_num);
  }
  code << "#else\n";
  if (input_data_type == kNumberTypeFloat32) {
    code.CodeFunction("Float32ToFp16", input_tensor_, output_tensor_, data_num);
  } else {
    MS_LOG(ERROR) << "Unsupported datatype from " << input_data_type << " cast to " << output_data_type;
    return RET_ERROR;
  }
  code << "#endif\n";
  context->AppendCode(code.str());
  return RET_OK;
}

int DTypeCastCoder::CastToOtherType(CoderContext *const context, TypeId input_data_type, TypeId output_data_type,
                                    const int data_num) {
  Serializer code;
  if (cast_to_other_func.find(std::make_pair(input_data_type, output_data_type)) == cast_to_other_func.end()) {
    MS_LOG(ERROR) << "Unsupported input or output data type, input data type " << input_data_type
                  << ", output data type " << output_data_type;
    return RET_ERROR;
  } else {
    code.CodeFunction(cast_to_other_func[{input_data_type, output_data_type}], input_tensor_, output_tensor_, data_num);
  }
  context->AppendCode(code.str());
  return RET_OK;
}

int DTypeCastCoder::DoCode(CoderContext *const context) {
  int data_num = MSMIN(stride_, data_num_ - kDefaultTaskId * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }
  TypeId input_data_type = input_tensor_->data_type();
  TypeId output_data_type = output_tensor_->data_type();
  Serializer code;
  if (input_data_type == output_data_type) {
    auto datalen = DataTypeSize(input_data_type);
    code.CodeFunction("memcpy", output_tensor_, input_tensor_, data_num * datalen);
    context->AppendCode(code.str());
    return RET_OK;
  }

  Collect(context,
          {
            "nnacl/base/cast_base.h",
          },
          {
            "cast_base.c",
            "common_func.c",
          });
  if (target_ == kARM32) {
    Collect(context, {}, {},
            {
              "PostFuncBiasReluC8.S",
              "PostFuncBiasReluC4.S",
            });
  } else if (target_ == kARM64) {
    Collect(context, {}, {},
            {
              "PostFuncBiasReluC8.S",
              "PostFuncBiasReluC4.S",
            });
  }
  if (output_data_type == kNumberTypeFloat32) {
    return CastToFloat32(context, input_data_type, output_data_type, data_num);
  } else if (output_data_type == kNumberTypeFloat16) {
    return CastToFloat16(context, input_data_type, output_data_type, data_num);
  } else {
    return CastToOtherType(context, input_data_type, output_data_type, data_num);
  }
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeUInt8, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt64, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeBool, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
}  // namespace mindspore::lite::micro
