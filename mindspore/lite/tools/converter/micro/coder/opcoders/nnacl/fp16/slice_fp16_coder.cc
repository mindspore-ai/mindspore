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

#include "mindspore/lite/tools/converter/micro/coder/opcoders/nnacl/fp16/slice_fp16_coder.h"
#include "mindspore/lite/tools/converter/micro/coder/opcoders/file_collector.h"
#include "mindspore/lite/tools/converter/micro/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore::lite::micro::nnacl {
int SliceFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/base/slice_base.h",
          },
          {
            "slice_base.c",
          });
  NNaclFp32Serializer code;
  code.CodeStruct("slice_struct", slice_struct_);
  code.CodeFunction("DoSliceNoParallel", input_tensor_, output_tensor_, "&slice_struct", slice_struct_.data_type_size_);
  context->AppendCode(code.str());
  return NNACL_OK;
}
int SliceFP16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16 || output_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  MS_CHECK_TRUE_MSG(SliceFP32Coder::Prepare(context) == RET_OK, RET_ERROR, "prepare slice fp16 coder failed!");
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_SliceFusion, CPUOpCoderCreator<SliceFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_SliceFusion, CPUOpCoderCreator<SliceFP16Coder>)
};  // namespace mindspore::lite::micro::nnacl
