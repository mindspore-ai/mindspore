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

#include "coder/opcoders/nnacl/int8/fullconnection_int8_coder.h"
#include "nnacl/int8/matmul_int8.h"
#include "coder/log.h"

using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::lite::micro::nnacl {
int FullConnectionInt8Coder::ReSize(CoderContext *const context) {
  int row = 1;
  for (size_t i = 0; i < output_tensor_->shape().size() - 1; ++i) {
    row *= (output_tensor_->shape()).at(i);
  }
  param_->row_ = row;
  param_->col_ = output_tensor_->shape().back();
  param_->deep_ = (filter_tensor_->shape()).at(1);
  MS_CHECK_RET_CODE(MatMulBaseInt8Coder::ReSize(context), "MatMulBaseInt8Coder::ReSize is nullptr");
  return RET_OK;
}

int FullConnectionInt8Coder::Prepare(CoderContext *const context) {
  // only support one thread currently
  thread_count_ = thread_num_;
  param_ = reinterpret_cast<MatMulParameter *>(parameter_);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data_c());
  }
  param_->batch = 1;
  param_->a_transpose_ = false;
  param_->b_transpose_ = true;
  MatMulBaseInt8Coder::InitParameter();
  MS_CHECK_RET_CODE(MatMulBaseInt8Coder::Init(), "Init failed");
  return ReSize(context);
}

int FullConnectionInt8Coder::DoCode(CoderContext *const context) {
  MS_CHECK_RET_CODE(MatMulBaseInt8Coder::DoCode(context), "matmul int8 do code failed");
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_FullConnection,
                   CPUOpCoderCreator<FullConnectionInt8Coder>)

}  // namespace mindspore::lite::micro::nnacl
