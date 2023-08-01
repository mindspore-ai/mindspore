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

#include "coder/opcoders/base/reshape_dynamic_base_coder.h"
#include <string>
#include "coder/opcoders/serializers/serializer.h"
#include "include/errorcode.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_ExpandDims;
using mindspore::schema::PrimitiveType_Flatten;
using mindspore::schema::PrimitiveType_FlattenGrad;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::lite::micro {
int ReshapeDynamicBaseCoder::Prepare(CoderContext *const context) {
  MS_CHECK_TRUE_MSG(input_tensors_.size() == C2NUM, RET_ERROR, "Reshape's input-num must be 2.");
  MS_CHECK_TRUE_MSG(input_tensors_[FIRST_INPUT]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Input tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->IsConst(), RET_NOT_SUPPORT,
                    "Currently, only support the first input of reshape is non-const when shape is dynamic.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->data_type() == kNumberTypeInt32 ||
                      input_tensors_[SECOND_INPUT]->data_type() == kNumberTypeInt,
                    RET_ERROR, "The data-type of Reshape's second input must be int.");
  MS_CHECK_TRUE_MSG(output_tensor_->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Output tensor data type is invalid.");
  return RET_OK;
}

int ReshapeDynamicBaseCoder::DoCode(CoderContext *const context) {
  Serializer coder;

  int data_item_size = static_cast<int>(lite::DataTypeSize(input_tensor_->data_type()));
  auto in_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  std::string size = AccumulateShape(in_shape) + " * " + std::to_string(data_item_size);
  std::string input_data = dynamic_mem_manager_->GetVarTensorAddr(input_tensor_);
  MS_CHECK_TRUE_MSG(!input_data.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  std::string output_data = dynamic_mem_manager_->GetVarTensorAddr(output_tensor_);
  MS_CHECK_TRUE_MSG(!output_data.empty(), RET_ERROR, "pointer is not allocated by the allocator");
  coder.CodeFunction("memcpy", output_data, input_data, size);

  context->AppendCode(coder.str());
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Reshape,
                           CPUOpCoderCreator<ReshapeDynamicBaseCoder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_ExpandDims,
                           CPUOpCoderCreator<ReshapeDynamicBaseCoder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Squeeze,
                           CPUOpCoderCreator<ReshapeDynamicBaseCoder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Unsqueeze,
                           CPUOpCoderCreator<ReshapeDynamicBaseCoder>)
}  // namespace mindspore::lite::micro
