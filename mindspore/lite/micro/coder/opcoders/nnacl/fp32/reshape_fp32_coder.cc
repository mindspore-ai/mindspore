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

#include "coder/opcoders/nnacl/fp32/reshape_fp32_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Reshape;
namespace mindspore::lite::micro::nnacl {

int ReshapeFP32Coder::DoCode(CoderContext *const context) {
  size_t data_size = input_tensor_->Size();

  Collect(context, {"nnacl/reshape.h"}, {"reshape.c"});

  NNaclFp32Serializer code;
  code.CodeFunction("Reshape", input_tensor_, output_tensor_, data_size);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Reshape, CPUOpCoderCreator<ReshapeFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Reshape, CPUOpCoderCreator<ReshapeFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
