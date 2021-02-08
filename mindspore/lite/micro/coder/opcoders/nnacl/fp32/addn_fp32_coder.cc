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

#include "micro/coder/opcoders/nnacl/fp32/addn_fp32_coder.h"
#include <string>
#include "micro/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "micro/coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_AddN;
namespace mindspore::lite::micro::nnacl {

int AddNFP32Coder::DoCode(CoderContext *const context) {
  Tensor *input0 = input_tensors_.at(kInputIndex);
  Tensor *input1 = input_tensors_.at(1);
  int elements_num = input0->ElementsNum();

  // Get Tensor Pointer
  std::string input0_str = allocator_->GetRuntimeAddr(input0);
  std::string input1_str = allocator_->GetRuntimeAddr(input1);
  Collect(context, {"nnacl/kernel/fp32/add_fp32_slim.h"}, {"add_fp32_slim.c"});
  NNaclFp32Serializer code;
  code.CodeFunction("ElementAdd", input0_str, input1_str, output_tensor_, elements_num);
  if (input_tensors_.size() > 2) {
    for (size_t i = 2; i < input_tensors_.size(); ++i) {
      std::string input_str = allocator_->GetRuntimeAddr(input_tensors_.at(i));
      code.CodeFunction("ElementAdd", input_str, output_tensor_, elements_num);
    }
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_AddN, CPUOpCoderCreator<AddNFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
