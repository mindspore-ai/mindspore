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

#include "src/litert/kernel/cpu/fp16/lstm_non_mindir_fp16.h"
#include "nnacl/fp16/lstm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kGateNum = 4;
constexpr size_t kInputTensorNumMin = 6;
}  // namespace

int LstmNonMindirFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputTensorNumMin);
  running_pack_ = train_mode_;
  for (size_t i = 1; i <= FOURTH_INPUT; ++i) {
    running_pack_ = running_pack_ || !in_tensors_[i]->IsConst();
  }
  return LstmFp16BaseCPUKernel::Prepare();
}

int LstmNonMindirFp16CPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  auto weight_i = in_tensors_.at(1);
  auto weight_i_data = weight_i->data();
  CHECK_NULL_RETURN(weight_i_data);
  weight_i_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_segment_num_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(weight_i_ptr_ != nullptr, lite::RET_NULL_PTR,
                    "LstmNonMindirCPUKernel malloc weight_i_ptr_ failed.");
  pack_buffer_.push_back(weight_i_ptr_);
  if (weight_i->data_type() == kNumberTypeFloat32) {
    PackLstmWeightFp32ToFp16(weight_i_ptr_, reinterpret_cast<float *>(weight_i_data), weight_segment_num_,
                             lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_);
  } else if (weight_i->data_type() == kNumberTypeFloat16) {
    PackLstmWeightFp16(weight_i_ptr_, reinterpret_cast<float16_t *>(weight_i_data), weight_segment_num_,
                       lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight_i tensor for lstm.";
    return RET_ERROR;
  }

  // input bias
  auto bias = in_tensors_.at(FOURTH_INPUT);
  auto bias_data = bias->data();
  CHECK_NULL_RETURN(bias_data);
  input_bias_ =
    reinterpret_cast<float16_t *>(malloc(weight_segment_num_ * lstm_param_->input_col_align_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(input_bias_ != nullptr, lite::RET_NULL_PTR, "LstmNonMindirCPUKernel malloc input_bias_ failed.");
  pack_buffer_.push_back(input_bias_);
  (void)memset(input_bias_, 0, weight_segment_num_ * lstm_param_->input_col_align_ * sizeof(float16_t));
  if (bias->data_type() == kNumberTypeFloat32) {
    PackLstmBiasFp32ToFp16(input_bias_, reinterpret_cast<float *>(bias_data), weight_segment_num_,
                           lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_);
  } else if (bias->data_type() == kNumberTypeFloat16) {
    PackLstmBiasFp16(input_bias_, reinterpret_cast<float16_t *>(bias_data), weight_segment_num_,
                     lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmNonMindirFp16CPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_h = in_tensors_.at(THIRD_INPUT);
  auto weight_h_data = weight_h->data();
  CHECK_NULL_RETURN(weight_h_data);
  weight_h_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_segment_num_ * lstm_param_->state_col_align_ * lstm_param_->output_size_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(weight_h_ptr_ != nullptr, lite::RET_NULL_PTR,
                    "LstmNonMindirCPUKernel malloc weight_h_ptr_ failed.");

  if (lstm_param_->batch_ != 1) {
    if (weight_h->data_type() == kNumberTypeFloat32) {
      PackLstmWeightFp32ToFp16(weight_h_ptr_, reinterpret_cast<float *>(weight_h_data), weight_segment_num_,
                               lstm_param_->output_size_, lstm_param_->hidden_size_, lstm_param_->state_col_align_);
    } else if (weight_h->data_type() == kNumberTypeFloat16) {
      PackLstmWeightFp16(weight_h_ptr_, reinterpret_cast<float16_t *>(weight_h_data), weight_segment_num_,
                         lstm_param_->output_size_, lstm_param_->hidden_size_, lstm_param_->state_col_align_);
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_h tensor for lstm.";
      return RET_ERROR;
    }
  } else {
    if (weight_h->data_type() == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<float *>(weight_h_data), weight_h_ptr_, weight_h->ElementsNum());
    } else if (weight_h->data_type() == kNumberTypeFloat16) {
      (void)memcpy(weight_h_ptr_, reinterpret_cast<float16_t *>(weight_h_data), weight_h->Size());
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_h tensor for lstm.";
      return RET_ERROR;
    }
  }

  // state bias
  auto bias = in_tensors_[FOURTH_INPUT];
  auto bias_data = bias->data();
  CHECK_NULL_RETURN(bias_data);
  state_bias_ =
    reinterpret_cast<float16_t *>(malloc(weight_segment_num_ * lstm_param_->state_col_align_ * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(state_bias_ != nullptr, lite::RET_NULL_PTR, "LstmNonMindirCPUKernel malloc state_bias_ failed.");
  (void)memset(state_bias_, 0, weight_segment_num_ * lstm_param_->state_col_align_ * sizeof(float16_t));
  if (bias->data_type() == kNumberTypeFloat32) {
    auto state_bias_data = reinterpret_cast<float *>(bias_data) + kGateNum * lstm_param_->hidden_size_;
    PackLstmBiasFp32ToFp16(state_bias_, state_bias_data, weight_segment_num_, lstm_param_->hidden_size_,
                           lstm_param_->state_col_align_, lstm_param_->bidirectional_);
  } else if (bias->data_type() == kNumberTypeFloat16) {
    auto state_bias_data = reinterpret_cast<float16_t *>(bias_data) + kGateNum * lstm_param_->hidden_size_;
    PackLstmBiasFp16(state_bias_, state_bias_data, weight_segment_num_, lstm_param_->hidden_size_,
                     lstm_param_->state_col_align_, lstm_param_->bidirectional_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmNonMindirFp16CPUKernel::InitProjectWeight() {
  if (in_tensors_.size() < C7NUM) {
    return RET_OK;
  }
  auto weight_pro = in_tensors_[SEVENTH_INPUT];
  auto shape = weight_pro->shape();
  MS_CHECK_TRUE_MSG(shape.size() == C3NUM, lite::RET_ERROR, "Project-weight's shape must be 3D.");
  auto weight_pro_data = weight_pro->data();
  CHECK_NULL_RETURN(weight_pro_data);
  int batch = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  if (shape[0] != batch) {
    MS_LOG(ERROR) << "Project-weight's shape[0] must be 1(bidirectional=false) or 2(bidirectional=true).";
    return RET_ERROR;
  }
  int pro_col_align = lstm_param_->batch_ == 1 ? lstm_param_->output_size_ : UP_ROUND(lstm_param_->output_size_, C8NUM);
  weight_project_ptr_ =
    reinterpret_cast<float16_t *>(malloc(batch * lstm_param_->hidden_size_ * pro_col_align * sizeof(float16_t)));
  MS_CHECK_TRUE_MSG(weight_project_ptr_ != nullptr, lite::RET_NULL_PTR,
                    "LstmNonMindirCPUKernel malloc weight_project_ptr_ failed.");

  if (lstm_param_->batch_ != 1) {
    if (weight_pro->data_type() == kNumberTypeFloat32) {
      PackLstmWeightFp32ToFp16(weight_project_ptr_, reinterpret_cast<float *>(weight_pro_data), batch,
                               lstm_param_->hidden_size_, lstm_param_->output_size_, pro_col_align);
    } else if (weight_pro->data_type() == kNumberTypeFloat16) {
      PackLstmWeightFp16(weight_project_ptr_, reinterpret_cast<float16_t *>(weight_pro_data), batch,
                         lstm_param_->hidden_size_, lstm_param_->output_size_, pro_col_align);
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_project tensor for lstm.";
      return RET_ERROR;
    }
  } else {
    if (weight_pro->data_type() == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<float *>(weight_pro_data), weight_project_ptr_, weight_pro->ElementsNum());
    } else if (weight_pro->data_type() == kNumberTypeFloat16) {
      (void)memcpy(weight_project_ptr_, weight_pro_data, weight_pro->Size());
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_project tensor for lstm.";
      return RET_ERROR;
    }
  }
  size_t bias_size = UP_ROUND(lstm_param_->output_size_, C8NUM) * sizeof(float16_t);
  project_bias_ = reinterpret_cast<float16_t *>(malloc(bias_size));
  MS_CHECK_TRUE_MSG(project_bias_ != nullptr, lite::RET_NULL_PTR,
                    "LstmNonMindirCPUKernel malloc project_bias_ failed.");
  (void)memset(project_bias_, 0, bias_size);
  return RET_OK;
}
}  // namespace mindspore::kernel
