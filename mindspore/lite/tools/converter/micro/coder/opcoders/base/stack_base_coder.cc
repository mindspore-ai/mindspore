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
#include "coder/opcoders/base/stack_base_coder.h"
#include <string>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_Stack;

namespace mindspore::lite::micro::nnacl {
int StackFP32Coder::Prepare(CoderContext *const context) {
  stack_param_ = reinterpret_cast<StackParameter *>(parameter_);
  return ReSize();
}

int StackFP32Coder::ReSize() {
  axis_ = stack_param_->axis_ >= 0 ? stack_param_->axis_
                                   : static_cast<int>(input_tensor_->shape().size()) + stack_param_->axis_ + 1;
  if (axis_ < 0 || axis_ > static_cast<int>(input_tensor_->shape().size())) {
    return RET_ERROR;
  }
  return RET_OK;
}

int StackFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/base/stack_base.h",
          },
          {
            "stack_base.c",
          });

  size_t input_num = input_tensors_.size();

  NNaclFp32Serializer code;
  code << "\t\tvoid *inputs_addr[] = {";
  for (size_t i = 0; i < input_num; ++i) {
    code << allocator_->GetRuntimeAddr(input_tensors_.at(i)) << ", ";
  }
  code << "};\n";

  size_t copy_size = 0;
  int outer_size = 1;
  auto shape = input_tensor_->shape();
  if (input_tensors_.empty()) {
    copy_size = 0;
    outer_size = 0;
  } else if (input_tensors_.size() == 1) {
    copy_size = input_tensor_->ElementsNum();
    outer_size = 1;
  } else {
    copy_size = 1;
    for (int i = axis_; i < static_cast<int>(shape.size()); ++i) {
      copy_size *= shape[i];
    }
    for (int i = 0; i < axis_; ++i) {
      outer_size *= shape[i];
    }
  }
  copy_size *= DataTypeSize(input_tensor_->data_type());
  code.CodeFunction("Stack", "inputs_addr", output_tensor_, input_num, copy_size, 0, outer_size);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Stack, CPUOpCoderCreator<StackFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Stack, CPUOpCoderCreator<StackFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
