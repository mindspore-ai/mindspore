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

#include "coder/opcoders/nnacl/fp16/transpose_dynamic_fp16_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_Transpose;
namespace mindspore::lite::micro::nnacl {
int TransposeDynamicFp16Coder::Prepare(CoderContext *const context) {
  MS_CHECK_TRUE_MSG(input_tensor_->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Input tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->data_type() == kNumberTypeInt32, RET_INPUT_PARAM_INVALID,
                    "Perm tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(
    output_tensor_->data_type() == kNumberTypeInt32 || output_tensor_->data_type() == kNumberTypeFloat16,
    RET_INPUT_PARAM_INVALID, "Output tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->IsConst(), RET_NOT_SUPPORT,
                    "The second input of transpose is non-const.");
  thread_num_ = 1;
  MS_CHECK_RET_CODE(Init(), "init failed");
  return RET_OK;
}

int TransposeDynamicFp16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/transpose_parameter.h",
            "nnacl/errorcode.h",
            "nnacl/fp16/transpose_fp16.h",
          },
          {
            "transpose_fp16.c",
          });

  NNaclFp32Serializer code;
  dims_ = static_cast<int>(out_shapes_.size());
  code << "const int32_t output_shape = [" << dims_ << "] = {";
  for (size_t i = 0; i < out_shapes_.size(); ++i) {
    code << out_shapes_[i] << ", ";
  }
  code << "};\n";
  code.CodeStruct("trans_param", *param_, dynamic_param_);
  auto input_str = dynamic_mem_manager_->GetVarTensorAddr(input_tensor_);
  auto output_str = dynamic_mem_manager_->GetVarTensorAddr(output_tensor_);
  if (param_->num_axes_ > DIMENSION_6D) {
    code.CodeFunction("TransposeDimsFp16", input_str, output_str, "output_shape", "trans_param.perm_",
                      "trans_param.strides_", "trans_param.out_strides_", "trans_param.num_axes_", kDefaultTaskId,
                      kDefaultThreadNum);
  } else {
    code.CodeFunction("DoTransposeFp16", input_str, output_str, "output_shape", "trans_param.perm_",
                      "trans_param.strides_", "trans_param.out_strides_", "trans_param.data_num_",
                      "trans_param.num_axes_");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_Transpose,
                           CPUOpCoderCreator<TransposeDynamicFp16Coder>)
}  // namespace mindspore::lite::micro::nnacl
