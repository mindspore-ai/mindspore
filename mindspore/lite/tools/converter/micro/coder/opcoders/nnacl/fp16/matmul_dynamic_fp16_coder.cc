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

#include "coder/opcoders/nnacl/fp16/matmul_dynamic_fp16_coder.h"
#include <vector>
#include "coder/opcoders/file_collector.h"
#include "coder/utils/coder_utils.h"

using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::lite::micro::nnacl {
int MatMulDynamicFP16Coder::InitAShape() {
  auto a_shape = shape_info_container_->GetTemplateShape(input_tensor_);
  MS_CHECK_TRUE_MSG(a_shape.size() >= DIMENSION_2D, RET_INPUT_PARAM_INVALID,
                    "MatMul's a_shape_size must be not less than two.");
  dynamic_params_.batch_ = AccumulateShape(a_shape);
  dynamic_params_.row_ = params_.a_transpose_ ? a_shape[a_shape.size() - C1NUM] : a_shape[a_shape.size() - C2NUM];
  return RET_OK;
}

int MatMulDynamicFP16Coder::InitBShape() {
  std::vector<int> b_shape = filter_tensor_->shape();
  MS_CHECK_TRUE_MSG(b_shape.size() >= DIMENSION_2D, RET_NOT_SUPPORT,
                    "MatMul's b_shape_size must be not less than two.");
  int batch = 1;
  for (size_t i = 0; i < b_shape.size() - DIMENSION_2D; ++i) {
    batch *= b_shape[i];
  }
  MS_CHECK_TRUE_MSG(batch == C1NUM, RET_NOT_SUPPORT, "Currently, MatMul only support matrix_B's batch is 1.");
  b_batch_ = batch;
  params_.col_ = params_.b_transpose_ ? b_shape[b_shape.size() - C2NUM] : b_shape[b_shape.size() - C1NUM];
  params_.col_8_ = UP_ROUND(params_.col_, C8NUM);
  params_.deep_ = params_.b_transpose_ ? b_shape[b_shape.size() - C1NUM] : b_shape[b_shape.size() - C2NUM];
  return RET_OK;
}

int MatMulDynamicFP16Coder::Prepare(CoderContext *const context) {
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_[i]->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                      "Input tensor data type is invalid.");
  }
  MS_CHECK_TRUE_MSG(output_tensor_->data_type() == kNumberTypeFloat16, RET_INPUT_PARAM_INVALID,
                    "Input tensor data type is invalid.");
  MS_CHECK_TRUE_MSG(input_tensors_.size() == C2NUM || input_tensors_.size() == C3NUM, RET_INPUT_PARAM_INVALID,
                    "MatMul's input-num must be 2 or 3.");
  MS_CHECK_TRUE_MSG(input_tensors_[SECOND_INPUT]->IsConst(), RET_NOT_SUPPORT,
                    "Currently, only support the first input of matmul is non-const when shape is dynamic.");
  if (input_tensors_.size() == C3NUM) {
    MS_CHECK_TRUE_MSG(input_tensors_[THIRD_INPUT]->IsConst(), RET_NOT_SUPPORT,
                      "Currently, only support the first input of matmul is non-const when shape is dynamic.");
  }
  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(parameter_);
  params_.act_type_ = matmul_param->act_type_;
  params_.thread_num_ = matmul_param->op_parameter_.thread_num_;
  params_.a_transpose_ = matmul_param->a_transpose_;
  params_.b_transpose_ = matmul_param->b_transpose_;
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data());
  }
  params_.a_const_ = (input_tensor_->data() != nullptr);
  params_.b_const_ = (filter_tensor_->data() != nullptr);
  MS_CHECK_RET_CODE(MatMulDynamicFP16BaseCoder::Prepare(context), "MatMulDynamicFP16Coder prepare failed");
  return RET_OK;
}

int MatMulDynamicFP16Coder::DoCode(CoderContext *const context) { return MatMulDynamicFP16BaseCoder::DoCode(context); }

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_MatMulFusion,
                           CPUOpCoderCreator<MatMulDynamicFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
