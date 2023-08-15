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

#include "coder/opcoders/nnacl/fp16/matmul_fp16_coder.h"
#include <vector>
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::lite::micro::nnacl {
int MatMulFP16Coder::InitAShape() {
  std::vector<int> a_shape = input_tensor_->shape();
  MS_CHECK_TRUE_MSG(a_shape.size() >= DIMENSION_2D, RET_ERROR, "A-metric tensor's shape is invalid.");
  int batch = 1;
  for (size_t i = 0; i < a_shape.size() - DIMENSION_2D; ++i) {
    batch *= a_shape.at(i);
  }
  params_.a_batch_ = batch;
  params_.row_ = params_.a_transpose_ ? a_shape[a_shape.size() - C1NUM] : a_shape[a_shape.size() - C2NUM];
  params_.deep_ = params_.a_transpose_ ? a_shape[a_shape.size() - C2NUM] : a_shape[a_shape.size() - C1NUM];
  params_.row_16_ = UP_ROUND(params_.row_, row_tile_);
  return RET_OK;
}

int MatMulFP16Coder::InitBShape() {
  std::vector<int> b_shape = filter_tensor_->shape();
  MS_CHECK_TRUE_MSG(b_shape.size() >= DIMENSION_2D, RET_ERROR, "B-metric tensor's shape is invalid.");
  int batch = 1;
  for (size_t i = 0; i < b_shape.size() - DIMENSION_2D; ++i) {
    batch *= b_shape[i];
  }
  params_.b_batch_ = batch;
  params_.col_ = params_.b_transpose_ ? b_shape[b_shape.size() - C2NUM] : b_shape[b_shape.size() - C1NUM];
  params_.col_8_ = UP_ROUND(params_.col_, C8NUM);
  params_.deep_ = params_.b_transpose_ ? b_shape[b_shape.size() - C1NUM] : b_shape[b_shape.size() - C2NUM];
  return RET_OK;
}

int MatMulFP16Coder::Prepare(CoderContext *const context) {
  CalculateOutBatchSize();
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(INFO) << "Input tensor data type is invalid";
    return RET_INPUT_PARAM_INVALID;
  }
  MatMulParameter *matmul_param = reinterpret_cast<MatMulParameter *>(parameter_);
  params_.act_type_ = matmul_param->act_type_;
  params_.thread_num_ = matmul_param->op_parameter_.thread_num_;
  params_.a_transpose_ = matmul_param->a_transpose_;
  params_.b_transpose_ = matmul_param->b_transpose_;
  MS_CHECK_TRUE_RET(input_tensors_.size() >= kBiasIndex, RET_ERROR);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data());
  }
  params_.a_const_ = (input_tensor_->data() != nullptr);
  params_.b_const_ = (filter_tensor_->data() != nullptr);
  MS_CHECK_RET_CODE(MatMulFP16BaseCoder::Prepare(context), "MatMulFP16Coder prepare failed");
  return RET_OK;
}

int MatMulFP16Coder::DoCode(CoderContext *const context) { return MatMulFP16BaseCoder::DoCode(context); }

REG_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_MatMulFusion, CPUOpCoderCreator<MatMulFP16Coder>)
REG_OPERATOR_CODER(kARM32, kNumberTypeFloat16, PrimitiveType_MatMulFusion, CPUOpCoderCreator<MatMulFP16Coder>)
}  // namespace mindspore::lite::micro::nnacl
