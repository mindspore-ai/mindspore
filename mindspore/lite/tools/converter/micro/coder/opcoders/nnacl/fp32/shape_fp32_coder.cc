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
#include "coder/opcoders/nnacl/fp32/shape_fp32_coder.h"
#include <string>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Shape;

namespace mindspore::lite::micro::nnacl {
int ShapeFP32Coder::DoCode(CoderContext *const context) {
  std::string output_str = allocator_->GetRuntimeAddr(output_tensor_);
  NNaclFp32Serializer code;
  MS_LOG(WARNING)
    << "The shape op can be fused and optimized by configuring the 'inputShape' parameter of the converter tool.";
  code << "  {\n";
  int index = 0;
  for (auto &shape : input_tensors_.at(0)->shape()) {
    code << "    " << output_str << "[" << index++ << "] = " << shape << ";\n";
  }
  code << "  }\n";

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Shape, CPUOpCoderCreator<ShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Shape, CPUOpCoderCreator<ShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeBool, PrimitiveType_Shape, CPUOpCoderCreator<ShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Shape, CPUOpCoderCreator<ShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeUInt8, PrimitiveType_Shape, CPUOpCoderCreator<ShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt64, PrimitiveType_Shape, CPUOpCoderCreator<ShapeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Shape, CPUOpCoderCreator<ShapeFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
