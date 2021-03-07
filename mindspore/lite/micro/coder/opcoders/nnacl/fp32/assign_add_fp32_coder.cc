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

#include "coder/opcoders/nnacl/fp32/assign_add_fp32_coder.h"
#include <string>
#include "schema/inner/ops_generated.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"

namespace mindspore::lite::micro::nnacl {

using mindspore::schema::PrimitiveType_AssignAdd;

int AssignAddFP32Coder::Prepare(CoderContext *const context) { return RET_OK; }

int AssignAddFP32Coder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(input_tensors_.size() == 2, "inputs size is not equal to two");
  Tensor *input0 = input_tensors_.at(0);
  Tensor *input1 = input_tensors_.at(1);
  if (input0->Size() != input1->Size()) {
    MS_LOG(ERROR) << "input0 size: " << input0->Size() << ", input1 size: " << input1->Size();
    return RET_ERROR;
  }

  NNaclFp32Serializer code;
  // Get Tensor Pointer
  std::string input0_str = allocator_->GetRuntimeAddr(input0);
  std::string input1_str = allocator_->GetRuntimeAddr(input1);
  size_t data_size = input0->Size();
  // assign add, just add input1'data to input0
  code << "\t\tfor (int i = 0; i < " << data_size << "; ++i) {\n";
  code << "\t\t\t(" << input0_str << ")[i] += (" << input1_str << ")[i];\n";
  code << "\t\t}\n";
  code.CodeFunction("memcpy", output_tensor_, input0_str, data_size);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_AssignAdd, CPUOpCoderCreator<AssignAddFP32Coder>)

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_AssignAdd, CPUOpCoderCreator<AssignAddFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
