/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/nnacl/fp16/concat_dynamic_fp16_coder.h"
#include <string>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::lite::micro::nnacl {
int ConcatDynamicFP16Coder::Prepare(CoderContext *const context) {
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_.at(i)->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "input tensor data type is invalid.");
  }
  concat_param_ = reinterpret_cast<ConcatParameter *>(parameter_);
  MS_CHECK_PTR(concat_param_);
  auto input_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  axis_ =
    concat_param_->axis_ >= 0 ? concat_param_->axis_ : static_cast<int>(input_shape.size()) + concat_param_->axis_;
  return RET_OK;
}

int ConcatDynamicFP16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/base/concat_base.h",
          },
          {
            "concat_base.c",
          });

  size_t input_num = input_tensors_.size();

  NNaclFp32Serializer code;
  code << "\t\tvoid *inputs_addr[] = {";
  for (size_t i = 0; i < input_num; ++i) {
    code << "(void *)("
         << GetTensorAddr(input_tensors_.at(i), input_tensors_.at(i)->IsConst(), dynamic_mem_manager_, allocator_)
         << "), ";
  }
  code << "};\n";

  size_t i;
  for (i = 0; i < input_num; ++i) {
    code << "\t\tint shape_" << i << "[] = {";
    auto in_shape = shape_info_container_->GetTemplateShape(input_tensors_.at(i));
    for (auto &shape : in_shape) {
      code << shape << ", ";
    }
    code << "};\n";
  }

  auto out_shape = shape_info_container_->GetTemplateShape(output_tensor_);
  code << "\t\tint shape_" << i << "[] = {";
  for (auto &shape : out_shape) {
    code << shape << ", ";
  }
  code << "};\n";

  code << "\t\tint *inputs_output_shape[] = {";
  for (i = 0; i <= input_num; ++i) {
    code << "shape_" << i << ", ";
  }
  code << "};\n";
  std::string output_data = GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  code.CodeFunction("Concat", "inputs_addr", input_num, axis_, "inputs_output_shape", out_shape.size(), output_data, 0,
                    1, sizeof(uint16_t));
  context->AppendCode(code.str());
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_Concat, CPUOpCoderCreator<ConcatDynamicFP16Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Concat, CPUOpCoderCreator<ConcatDynamicFP16Coder>)

}  // namespace mindspore::lite::micro::nnacl
