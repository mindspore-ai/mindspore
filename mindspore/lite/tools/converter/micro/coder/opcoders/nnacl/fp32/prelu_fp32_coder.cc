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
#include "coder/opcoders/nnacl/fp32/prelu_fp32_coder.h"
#include <string>
#include "nnacl/fp32/prelu_fp32.h"
#include "nnacl/op_base.h"
#include "coder/allocator/allocator.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_PReLUFusion;

namespace mindspore::lite::micro::nnacl {
int PReluFP32Coder::DoCode(CoderContext *const context) {
  int count = input_tensor_->ElementsNum();
  Collect(context,
          {
            "nnacl/fp32/prelu_fp32.h",
          },
          {
            "prelu_fp32.c",
          });
  NNaclFp32Serializer code;
  constexpr size_t kInputNum = 2;
  if (input_tensors_.size() != kInputNum) {
    return RET_ERROR;
  }
  if (input_tensors_[1]->ElementsNum() == 1) {
    std::string input1_data = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[1], true);
    if (input1_data.empty()) {
      MS_LOG(ERROR) << "pointer is not allocated by the allocator";
      return RET_ERROR;
    }
    input1_data = input1_data + "[0]";
    code.CodeFunction("PReluShareChannel", input_tensor_, output_tensor_, input1_data, 0, count);
  } else {
    code.CodeFunction("PRelu", input_tensor_, output_tensor_, input_tensors_[1], 0, count, input_tensor_->Channel());
  }

  MS_LOG(DEBUG) << "PReluFP32Coder has been called";
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_PReLUFusion, CPUOpCoderCreator<PReluFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
