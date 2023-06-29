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
#include "coder/opcoders/nnacl/fp16/layernorm_fp16_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::lite::micro::nnacl {
int LayerNormFP16Coder::Prepare(CoderContext *const context) {
  if ((input_tensor_->data_type() != kNumberTypeFloat16) ||
      (input_tensors_.at(SECOND_INPUT)->data_type() != kNumberTypeFloat16) ||
      (input_tensors_.at(THIRD_INPUT)->data_type() != kNumberTypeFloat16)) {
    MS_LOG(INFO) << "Input tensors data type is invalid";
    return RET_INPUT_PARAM_INVALID;
  }
  return LayerNormFP32Coder::Prepare(context);
}

int LayerNormFP16Coder::DoCode(CoderContext *const context) {
  NNaclFp32Serializer code;
  code.CodeStruct("layer_norm_compute_parm", compute_);
  Collect(context, {"nnacl/fp16/layer_norm_fp16.h"}, {"layer_norm_fp16.c"});
  if (output_tensors_.size() == C3NUM) {
    code.CodeFunction("LayerNormFp16", input_tensor_, input_tensors_.at(SECOND_INPUT), input_tensors_.at(THIRD_INPUT),
                      output_tensor_, output_tensors_.at(SECOND_INPUT), output_tensors_.at(THIRD_INPUT),
                      "&layer_norm_compute_parm", 0, 1);
  } else if (output_tensors_.size() == 1) {
    code.CodeFunction("LayerNormFp16", input_tensor_, input_tensors_.at(SECOND_INPUT), input_tensors_.at(THIRD_INPUT),
                      output_tensor_, "NULL", "NULL", "&layer_norm_compute_parm", 0, 1);
  } else {
    MS_LOG(ERROR) << "LayerNorm should have 1 or 3 output tensors";
    return RET_ERROR;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_LayerNormFusion, CPUOpCoderCreator<LayerNormFP16Coder>)
REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_LayerNormFusion, CPUOpCoderCreator<LayerNormFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
