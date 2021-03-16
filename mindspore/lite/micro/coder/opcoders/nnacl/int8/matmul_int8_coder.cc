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

#include "coder/opcoders/nnacl/int8/matmul_int8_coder.h"
#include "coder/opcoders/op_coder.h"
using mindspore::schema::PrimitiveType_MatMul;
namespace mindspore::lite::micro::nnacl {
int MatMulInt8Coder::Prepare(CoderContext *const context) {
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data_c());
  }
  param_ = reinterpret_cast<MatMulParameter *>(parameter_);
  MatMulBaseInt8Coder::InitParameter();
  MS_CHECK_RET_CODE(MatMulBaseInt8Coder::Init(), "ParallelLaunch failed");
  return ReSize(context);
}

int MatMulInt8Coder::ReSize(CoderContext *const context) {
  int batch = 1;
  std::vector<int> x_shape = input_tensor_->shape();
  std::vector<int> o_shape = output_tensor_->shape();
  MS_CHECK_RET_CODE(x_shape.size() >= kBiasIndex, "x_shape size is less than two");
  for (size_t i = 0; i < x_shape.size() - kBiasIndex; ++i) {
    batch *= x_shape[i];
  }
  param_->batch = batch;
  MS_CHECK_RET_CODE(o_shape.size() >= kBiasIndex, "o_shape size is less than two");
  param_->row_ = o_shape[o_shape.size() - kBiasIndex];
  param_->col_ = o_shape[o_shape.size() - kWeightIndex];
  param_->deep_ = param_->a_transpose_ ? x_shape[x_shape.size() - kBiasIndex] : x_shape[x_shape.size() - kWeightIndex];
  MS_CHECK_RET_CODE(MatMulBaseInt8Coder::ReSize(context), "MatMulBaseInt8Coder::ReSize is nullptr");
  return RET_OK;
}

int MatMulInt8Coder::DoCode(CoderContext *const context) {
  MS_CHECK_RET_CODE(MatMulBaseInt8Coder::DoCode(context), "matmul int8 do code failed");
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_MatMul, CPUOpCoderCreator<MatMulInt8Coder>)

}  // namespace mindspore::lite::micro::nnacl
