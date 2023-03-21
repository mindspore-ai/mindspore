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
#include "coder/opcoders/nnacl/fp32/layernorm_fp32_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_LayerNormFusion;

namespace mindspore::lite::micro::nnacl {
namespace {
constexpr size_t kOutputNum = 3;
}
int LayerNormFP32Coder::Prepare(CoderContext *const context) {
  param_ = reinterpret_cast<LayerNormParameter *>(parameter_);
  param_->op_parameter_.thread_num_ = 1;
  auto shape = input_tensor_->shape();
  param_->begin_norm_axis_ = param_->begin_norm_axis_ >= 0 ? param_->begin_norm_axis_
                                                           : param_->begin_norm_axis_ + static_cast<int>(shape.size());
  param_->begin_params_axis_ = param_->begin_params_axis_ >= 0
                                 ? param_->begin_params_axis_
                                 : param_->begin_params_axis_ + static_cast<int>(shape.size());
  MS_CHECK_LT(param_->begin_norm_axis_, static_cast<int>(shape.size()), RET_ERROR);
  MS_CHECK_LT(param_->begin_params_axis_, static_cast<int>(shape.size()), RET_ERROR);
  param_->norm_outer_size_ = 1;
  for (int i = 0; i < param_->begin_norm_axis_; ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->norm_outer_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->norm_outer_size_ *= shape.at(i);
  }
  param_->norm_inner_size_ = 1;
  for (size_t i = param_->begin_norm_axis_; i < shape.size(); ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->norm_inner_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->norm_inner_size_ *= shape.at(i);
  }
  param_->params_outer_size_ = 1;
  for (int i = 0; i < param_->begin_params_axis_; ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->params_outer_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->params_outer_size_ *= shape.at(i);
  }
  param_->params_inner_size_ = 1;
  for (size_t i = param_->begin_params_axis_; i < shape.size(); ++i) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param_->params_inner_size_, shape.at(i)), RET_ERROR, "mul overflow.");
    param_->params_inner_size_ *= shape.at(i);
  }
  return RET_OK;
}

int LayerNormFP32Coder::DoCode(CoderContext *const context) {
  NNaclFp32Serializer code;
  code.CodeStruct("layer_norm_parm", *param_);
  Collect(context, {"nnacl/fp32/layer_norm_fp32.h"}, {"layer_norm_fp32.c"});
  if (output_tensors_.size() == kOutputNum) {
    code.CodeFunction("LayerNorm", input_tensor_, input_tensors_.at(SECOND_INPUT), input_tensors_.at(THIRD_INPUT),
                      output_tensor_, output_tensors_.at(SECOND_INPUT), output_tensors_.at(THIRD_INPUT),
                      "&layer_norm_parm", 0);
  } else if (output_tensors_.size() == 1) {
    code.CodeFunction("LayerNorm", input_tensor_, input_tensors_.at(SECOND_INPUT), input_tensors_.at(THIRD_INPUT),
                      output_tensor_, "NULL", "NULL", "&layer_norm_parm", 0);
  } else {
    return RET_ERROR;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_LayerNormFusion,
                   CPUOpCoderCreator<LayerNormFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
