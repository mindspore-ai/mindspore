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

#include "coder/opcoders/nnacl/fp32/ones_like_fp32_coder.h"
#include <string>
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "include/errorcode.h"

using mindspore::schema::PrimitiveType_OnesLike;

namespace mindspore::lite::micro::nnacl {
int OnesLikeFP32Coder::Prepare(CoderContext *const context) { return RET_OK; }

int OnesLikeFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/kernel/ones_like.h",
          },
          {
            "ones_like.c",
          });

  NNaclFp32Serializer coder;
  std::string output_str = allocator_->GetRuntimeAddr(output_tensor_);
  coder << "for (size_t i = 0; i < " << output_tensor_->ElementsNum() << "; ++i) {\n"
        << output_str << "[i] = 1;\n"
        << "}\n";
  context->AppendCode(coder.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_OnesLike, CPUOpCoderCreator<OnesLikeFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_OnesLike, CPUOpCoderCreator<OnesLikeFP32Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_OnesLike, CPUOpCoderCreator<OnesLikeFP32Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_OnesLike, CPUOpCoderCreator<OnesLikeFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
