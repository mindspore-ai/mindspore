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

#include "coder/opcoders/nnacl/fp32/full_connection_fp32_coder.h"
#include "coder/log.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::lite::micro::nnacl {
int FullConnectionFP32Coder::ReSize() {
  int row = 1;
  for (int i = 0; i < static_cast<int>(output_tensor_->shape().size() - 1); ++i) {
    row *= output_tensor_->shape().at(i);
  }
  params_->row_ = row;
  params_->col_ = output_tensor_->shape().back();
  params_->deep_ = filter_tensor_->shape().at(1);
  return MatMulFP32BaseCoder::ReSize();
}

int FullConnectionFP32Coder::Init() {
  this->params_ = reinterpret_cast<MatMulParameter *>(parameter_);
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
    std::vector<int> a_shape = input_tensor_->shape();
    params_->row_ = a_shape.at(0);
    params_->deep_ = a_shape.at(1);
  }

  if (params_->b_const_) {
    std::vector<int> b_shape = filter_tensor_->shape();
    params_->col_ = b_shape.at(0);
    params_->deep_ = b_shape.at(1);
  }
  params_->batch = 1;
  params_->a_transpose_ = false;
  params_->b_transpose_ = true;
  MS_CHECK_RET_CODE(MatMulFP32BaseCoder::Init(), "MatMulFP32BaseCoder init failed");
  if (params_->row_ == 1 && !params_->b_const_) {
    vec_matmul_ = true;
  }
  return RET_OK;
}

int FullConnectionFP32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(Init(), "FullConnectionFP32Coder Init failed");
  return ReSize();
}

int FullConnectionFP32Coder::DoCode(CoderContext *const context) { return MatMulFP32BaseCoder::DoCode(context); }

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_FullConnection,
                   CPUOpCoderCreator<FullConnectionFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
