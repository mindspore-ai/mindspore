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

#include "coder/opcoders/base/reshape_base_coder.h"
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/serializers/serializer.h"
#include "include/errorcode.h"

using mindspore::schema::PrimitiveType_ExpandDims;
using mindspore::schema::PrimitiveType_Flatten;
using mindspore::schema::PrimitiveType_FlattenGrad;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::lite::micro {
int ReshapeBaseCoder::Prepare(CoderContext *const context) { return RET_OK; }

int ReshapeBaseCoder::DoCode(CoderContext *const context) {
  Serializer coder;

  size_t size = input_tensor_->Size();
  coder.CodeFunction("memcpy", output_tensor_, input_tensor_, size);

  context->AppendCode(coder.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Reshape, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Reshape, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Reshape, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Flatten, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Flatten, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Flatten, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_ExpandDims, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_ExpandDims, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_ExpandDims, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Squeeze, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Squeeze, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Squeeze, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Unsqueeze, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Unsqueeze, CPUOpCoderCreator<ReshapeBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Unsqueeze, CPUOpCoderCreator<ReshapeBaseCoder>)
}  // namespace mindspore::lite::micro
