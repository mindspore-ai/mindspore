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
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/base/dtype_cast_coder.h"
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Cast;
namespace mindspore::lite::micro {

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

int DTypeCastCoder::DoCode(CoderContext *const context) {
  int data_num = MSMIN(stride_, data_num_ - kDefaultTaskId * stride_);
  if (data_num <= 0) {
    return RET_OK;
  }
  TypeId input_data_type = input_tensor_->data_type();
  TypeId output_data_type = output_tensor_->data_type();

  std::vector<std::string> asmFiles;
  if (target_ == kARM32A) {
    asmFiles = {"nnacl/assembly/arm32/PostFuncBiasReluC8.S", "nnacl/assembly/arm32/PostFuncBiasReluC4.S"};
  } else if (target_ == kARM64) {
    asmFiles = {"nnacl/assembly/arm64/PostFuncBiasReluC8.S", "nnacl/assembly/arm64/PostFuncBiasReluC4.S"};
  }
  Collect(context, {"nnacl/fp32/cast.h"}, {"nnacl/fp32/cast.c", "nnacl/fp32/common_func.c"}, asmFiles);
  Serializer code;
  if (output_data_type != kNumberTypeFloat32) {
    if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt32) {
      std::string input_str = allocator_->GetRuntimeAddr(input_tensor_);
      std::string output_str = allocator_->GetRuntimeAddr(output_tensor_);
      code << "\t\tfor (int i = 0; i < " << data_num << "; ++i) {\n";
      code << "\t\t\t(" << output_str << ")[i] = (" << input_str << ")[i];\n";
      code << "\t\t}\n";
      context->AppendCode(code.str());
      return RET_OK;
    } else if (input_data_type != kNumberTypeFloat32 && output_data_type == kNumberTypeInt32) {
      code.CodeFunction("Float32ToInt32", input_tensor_, output_tensor_, data_num);
    } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeFloat16) {
      code.CodeFunction("Float32ToFp16", input_tensor_, output_tensor_, data_num);
    } else {
      MS_LOG(ERROR) << "Unsupported datatype from " << input_data_type << " to " << output_data_type;
      return RET_ERROR;
    }
  } else {
    switch (input_data_type) {
      case kNumberTypeUInt8:
        code.CodeFunction("Uint8ToFloat32", input_tensor_, output_tensor_, data_num);
        break;
      case kNumberTypeInt32:
        code.CodeFunction("Int32ToFloat32", input_tensor_, output_tensor_, data_num);
        break;
      case kNumberTypeFloat16:
        code.CodeFunction("Fp16ToFloat32", input_tensor_, output_tensor_, data_num);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported input data type " << input_data_type;
        return RET_ERROR;
    }
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeUInt8, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Cast, CPUOpCoderCreator<DTypeCastCoder>)
}  // namespace mindspore::lite::micro
