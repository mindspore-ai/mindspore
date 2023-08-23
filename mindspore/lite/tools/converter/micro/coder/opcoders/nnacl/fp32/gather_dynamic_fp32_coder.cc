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

#include "coder/opcoders/nnacl/fp32/gather_dynamic_fp32_coder.h"
#include <string>
#include "nnacl/gather_parameter.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::lite::micro::nnacl {
int GatherDynamicFP32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_TRUE_MSG(input_tensors_.size() == C3NUM, RET_ERROR, "Gather's input-num must be 3.");
  MS_CHECK_TRUE_MSG(input_tensors_[FIRST_INPUT]->IsConst() && input_tensors_[THIRD_INPUT]->IsConst(), RET_NOT_SUPPORT,
                    "Currently, only support the second input of gather is non-const when shape is dynamic.");
  MS_CHECK_TRUE_MSG(input_tensors_[THIRD_INPUT]->data_type() == kNumberTypeInt32 ||
                      input_tensors_[THIRD_INPUT]->data_type() == kNumberTypeInt,
                    RET_ERROR, "The data-type of Gather's third input must be int.");
  MS_CHECK_TRUE_MSG(input_tensors_[THIRD_INPUT]->data() != nullptr, RET_NULL_PTR, "Gather has no axis.");
  auto axis = input_tensors_[THIRD_INPUT]->data();
  axis_ = *(static_cast<int *>(axis));
  auto in_shape0 = input_tensors_[FIRST_INPUT]->shape();
  axis_ = axis_ >= 0 ? axis_ : axis_ + static_cast<int>(in_shape0.size());
  MS_CHECK_TRUE_MSG(axis_ >= 0 && axis_ < static_cast<int>(in_shape0.size()), RET_INPUT_TENSOR_ERROR,
                    "Gather's axis is out of range.");
  return RET_OK;
}

int GatherDynamicFP32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/base/gather_base.h",
          },
          {
            "gather_base.c",
          });
  auto in_shape0 = input_tensors_[FIRST_INPUT]->shape();
  auto data_item_size = static_cast<int>(lite::DataTypeSize(input_tensors_[FIRST_INPUT]->data_type()));
  int64_t out_size = 1;
  for (size_t i = 0; i < static_cast<size_t>(axis_); ++i) {
    out_size *= in_shape0[i];
  }
  int64_t byte_inner_size = data_item_size;
  for (size_t i = axis_ + 1; i < in_shape0.size(); ++i) {
    byte_inner_size *= in_shape0[i];
  }
  int64_t limit = in_shape0[axis_];
  auto in_shape1 = shape_info_container_->GetTemplateShape(input_tensors_[SECOND_INPUT]);
  std::string byte_out_stride_str = AccumulateShape(in_shape1) + " * " + std::to_string(byte_inner_size);
  std::string index_num_str = AccumulateShape(in_shape1);
  std::string input0_data = MemoryAllocator::GetInstance()->GetRuntimeAddr(input_tensors_[FIRST_INPUT], true);
  MS_CHECK_TRUE_MSG(!input0_data.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  std::string input1_data = dynamic_mem_manager_->GetVarTensorAddr(input_tensors_[SECOND_INPUT]);
  MS_CHECK_TRUE_MSG(!input1_data.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  std::string output_data = dynamic_mem_manager_->GetVarTensorAddr(output_tensors_[FIRST_INPUT]);
  MS_CHECK_TRUE_MSG(!output_data.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  NNaclFp32Serializer code;
  code << "\t\tconst int8_t *int8_in = (const int8_t *)(" << input0_data << ");\n";
  code << "\t\tconst int *index_data = (const int *)(" << input1_data << ");\n";
  code << "\t\tint8_t *int8_out = (int8_t *)(" << output_data << ");\n";
  code << "\t\tint error_index = -1;\n";
  // call the op function
  code.CodeFunction("Gather", "int8_in", out_size, byte_inner_size, limit, "index_data", index_num_str, "int8_out",
                    byte_out_stride_str, "&error_index");
  context->AppendCode(code.str());
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat32, PrimitiveType_Gather, CPUOpCoderCreator<GatherDynamicFP32Coder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Gather, CPUOpCoderCreator<GatherDynamicFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
