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

#include "coder/opcoders/nnacl/fp32/addn_fp32_coder.h"
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_AddN;
namespace mindspore::lite::micro::nnacl {

int AddNFP32Coder::DoCode(CoderContext *const context) {
  Tensor *input0 = input_tensors_.at(kInputIndex);
  Tensor *input1 = input_tensors_.at(1);
  int elements_num = input0->ElementsNum();

  // Get Tensor Pointer
  Collect(context, {"nnacl/kernel/fp32/add_fp32.h"}, {"add_fp32.c", "arithmetic_fp32.c", "arithmetic_base.c"});
  NNaclFp32Serializer code;
  code.CodeFunction("ElementAdd", input0, input1, output_tensor_, elements_num);
  if (input_tensors_.size() > 2) {
    for (size_t i = 2; i < input_tensors_.size(); ++i) {
      code.CodeFunction("ElementAdd", input_tensors_.at(i), output_tensor_, elements_num);
    }
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_AddN, CPUOpCoderCreator<AddNFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
