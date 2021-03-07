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

#include "coder/opcoders/nnacl/fp32/matmul_fp32_coder.h"
#include <vector>
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/nnacl/dequant/de_quant.h"

using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::lite::micro::nnacl {

int MatMulFP32Coder::InitShapeA() {
  std::vector<int> a_shape = input_tensor_->shape();
  int a_shape_size = static_cast<int>(a_shape.size());
  if (a_shape_size < kBiasIndex) {
    MS_LOG(ERROR) << "a_shape_size is less than two";
    return RET_ERROR;
  }
  int batch = 1;
  for (int i = 0; i < a_shape_size - 2; ++i) {
    batch *= a_shape.at(i);
  }
  params_->batch = batch;
  params_->row_ = params_->a_transpose_ ? a_shape.at(a_shape_size - 1) : a_shape.at(a_shape_size - 2);
  params_->deep_ = params_->a_transpose_ ? a_shape.at(a_shape_size - 2) : a_shape.at(a_shape_size - 1);
  return RET_OK;
}

int MatMulFP32Coder::InitShapeB() {
  std::vector<int> b_shape = filter_tensor_->shape();
  int b_shape_size = static_cast<int>(b_shape.size());
  if (b_shape_size < kBiasIndex) {
    MS_LOG(ERROR) << "a_shape_size is less than two";
    return RET_ERROR;
  }
  int batch = 1;
  for (int i = 0; i < b_shape_size - 2; ++i) {
    batch *= b_shape.at(i);
  }
  params_->batch = batch;
  params_->col_ = params_->b_transpose_ ? b_shape.at(b_shape_size - 2) : b_shape.at(b_shape_size - 1);
  params_->deep_ = params_->b_transpose_ ? b_shape.at(b_shape_size - 1) : b_shape.at(b_shape_size - 2);
  return RET_OK;
}

// this function is a temporary for inferShapeDone
int MatMulFP32Coder::ReSize() {
  MS_CHECK_RET_CODE(InitShapeA(), "MatMulFP32Coder init_shape_a failed");
  MS_CHECK_RET_CODE(InitShapeB(), "MatMulFP32Coder init_shape_b failed");
  return MatMulFP32BaseCoder::ReSize();
}

int MatMulFP32Coder::Prepare(CoderContext *const context) {
  params_ = reinterpret_cast<MatMulParameter *>(parameter_);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data_c());
  }
  params_->a_const_ = (input_tensor_->data_c() != nullptr);
  params_->b_const_ = (filter_tensor_->data_c() != nullptr);
  MatMulFP32BaseCoder::InitParameter();
  if (params_->a_const_) {
    de_quant_flag_ = Dequant::GetInstance()->CheckDequantFlag(input_tensor_);
    MS_CHECK_RET_CODE(InitShapeA(), "MatMulFP32Coder init_shape_a failed");
  }
  if (params_->b_const_) {
    de_quant_flag_ = Dequant::GetInstance()->CheckDequantFlag(filter_tensor_);
    MS_CHECK_RET_CODE(InitShapeB(), "MatMulFP32Coder init_shape_b failed");
  }
  MS_CHECK_RET_CODE(MatMulFP32BaseCoder::Init(), "MatMulFP32Coder init failed");
  return ReSize();
}

int MatMulFP32Coder::DoCode(CoderContext *const context) { return MatMulFP32BaseCoder::DoCode(context); }

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_MatMul, CPUOpCoderCreator<MatMulFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
