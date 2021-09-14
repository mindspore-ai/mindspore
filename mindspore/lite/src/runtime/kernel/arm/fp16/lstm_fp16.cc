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

#include "src/runtime/kernel/arm/fp16/lstm_fp16.h"
#include <vector>
#include <cfloat>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/lstm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LSTM;

namespace mindspore::kernel {
void LstmFp16CPUKernel::FreeTmpBuffer() {
  if (weight_i_ptr_ != nullptr) {
    free(weight_i_ptr_);
    weight_i_ptr_ = nullptr;
  }
  if (input_bias_ != nullptr) {
    free(input_bias_);
    input_bias_ = nullptr;
  }
  if (weight_h_ptr_ != nullptr) {
    free(weight_h_ptr_);
    weight_h_ptr_ = nullptr;
  }
  if (state_bias_ != nullptr) {
    free(state_bias_);
    state_bias_ = nullptr;
  }
}

void LstmFp16CPUKernel::FreeRunBuffer() {
  ms_context_->allocator->Free(buffer_[packed_input_index]);
  ms_context_->allocator->Free(buffer_[input_gate_index]);
  if (!is_vec_) {
    ms_context_->allocator->Free(buffer_[packed_state_index]);
  }
  ms_context_->allocator->Free(buffer_[state_gate_index]);
  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    ms_context_->allocator->Free(buffer_[cell_state_index]);
  }
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    ms_context_->allocator->Free(buffer_[hidden_state_index]);
  }
}

int LstmFp16CPUKernel::InitParam() {
  auto input = in_tensors_.front();
  std::vector<int> in_shape = input->shape();
  lstm_param_->seq_len_ = in_shape.at(0);
  lstm_param_->batch_ = in_shape.at(1);
  lstm_param_->input_size_ = in_shape.at(2);

  auto weight_i = in_tensors_.at(1);
  std::vector<int> w_shape = weight_i->shape();
  lstm_param_->hidden_size_ = w_shape.at(1) / gate_num;

  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? 2 * lstm_param_->batch_ * lstm_param_->hidden_size_
                                                          : lstm_param_->batch_ * lstm_param_->hidden_size_;
  weight_batch_ = lstm_param_->bidirectional_ ? 2 * gate_num : gate_num;
  lstm_param_->input_row_align_ = UP_ROUND(lstm_param_->seq_len_ * lstm_param_->batch_, C16NUM);
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, C8NUM);

  is_vec_ = lstm_param_->batch_ == 1;
  lstm_param_->state_row_align_ = is_vec_ ? lstm_param_->batch_ : UP_ROUND(lstm_param_->batch_, C16NUM);
  lstm_param_->state_col_align_ = is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, C8NUM);
  return RET_OK;
}

int LstmFp16CPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  auto weight_i = in_tensors_.at(1);
  auto weight_i_data = weight_i->data();
  CHECK_NULL_RETURN(weight_i_data);
  weight_i_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_batch_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float16_t)));
  if (weight_i_ptr_ == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc weight_i_ptr_ error.";
    return RET_ERROR;
  }
  if (weight_i->data_type() == kNumberTypeFloat32) {
    PackLstmWeightFp32ToFp16(weight_i_ptr_, reinterpret_cast<float *>(weight_i_data), weight_batch_,
                             lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_);
  } else if (weight_i->data_type() == kNumberTypeFloat16) {
    PackLstmWeightFp16(weight_i_ptr_, reinterpret_cast<float16_t *>(weight_i_data), weight_batch_,
                       lstm_param_->input_size_, lstm_param_->hidden_size_, lstm_param_->input_col_align_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight_i tensor for lstm.";
    return RET_ERROR;
  }

  // input bias
  auto bias = in_tensors_.at(3);
  auto bias_data = bias->data();
  CHECK_NULL_RETURN(bias_data);
  input_bias_ =
    reinterpret_cast<float16_t *>(malloc(weight_batch_ * lstm_param_->input_col_align_ * sizeof(float16_t)));
  if (input_bias_ == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc input_bias_ error.";
    return RET_ERROR;
  }
  memset(input_bias_, 0, weight_batch_ * lstm_param_->input_col_align_ * sizeof(float16_t));
  if (bias->data_type() == kNumberTypeFloat32) {
    PackLstmBiasFp32ToFp16(input_bias_, reinterpret_cast<float *>(bias_data), weight_batch_, lstm_param_->hidden_size_,
                           lstm_param_->input_col_align_, lstm_param_->bidirectional_);
  } else if (bias->data_type() == kNumberTypeFloat16) {
    PackLstmBiasFp16(input_bias_, reinterpret_cast<float16_t *>(bias_data), weight_batch_, lstm_param_->hidden_size_,
                     lstm_param_->input_col_align_, lstm_param_->bidirectional_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16CPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_h = in_tensors_.at(2);
  auto weight_h_data = weight_h->data();
  CHECK_NULL_RETURN(weight_h_data);
  weight_h_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_batch_ * lstm_param_->state_col_align_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
  if (weight_h_ptr_ == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc weight_h_ptr_ error.";
    return RET_ERROR;
  }

  if (!is_vec_) {
    if (weight_h->data_type() == kNumberTypeFloat32) {
      PackLstmWeightFp32ToFp16(weight_h_ptr_, reinterpret_cast<float *>(weight_h_data), weight_batch_,
                               lstm_param_->hidden_size_, lstm_param_->hidden_size_, lstm_param_->state_col_align_);
    } else if (weight_h->data_type() == kNumberTypeFloat16) {
      PackLstmWeightFp16(weight_h_ptr_, reinterpret_cast<float16_t *>(weight_h_data), weight_batch_,
                         lstm_param_->hidden_size_, lstm_param_->hidden_size_, lstm_param_->state_col_align_);
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_h tensor for lstm.";
      return RET_ERROR;
    }
  } else {
    if (weight_h->data_type() == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<float *>(weight_h_data), weight_h_ptr_, weight_h->ElementsNum());
    } else if (weight_h->data_type() == kNumberTypeFloat16) {
      memcpy(weight_h_ptr_, reinterpret_cast<float16_t *>(weight_h_data), weight_h->Size());
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_h tensor for lstm.";
      return RET_ERROR;
    }
  }

  // state bias
  auto bias = in_tensors_.at(3);
  auto bias_data = bias->data();
  CHECK_NULL_RETURN(bias_data);
  state_bias_ =
    reinterpret_cast<float16_t *>(malloc(weight_batch_ * lstm_param_->state_col_align_ * sizeof(float16_t)));
  if (state_bias_ == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state_bias_ error.";
    return RET_ERROR;
  }
  memset(state_bias_, 0, weight_batch_ * lstm_param_->state_col_align_ * sizeof(float16_t));
  if (bias->data_type() == kNumberTypeFloat32) {
    auto state_bias_data = reinterpret_cast<float *>(bias_data) + gate_num * lstm_param_->hidden_size_;
    PackLstmBiasFp32ToFp16(state_bias_, state_bias_data, weight_batch_, lstm_param_->hidden_size_,
                           lstm_param_->state_col_align_, lstm_param_->bidirectional_);
  } else if (bias->data_type() == kNumberTypeFloat16) {
    auto state_bias_data = reinterpret_cast<float16_t *>(bias_data) + gate_num * lstm_param_->hidden_size_;
    PackLstmBiasFp16(state_bias_, state_bias_data, weight_batch_, lstm_param_->hidden_size_,
                     lstm_param_->state_col_align_, lstm_param_->bidirectional_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 6);
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    CHECK_NULL_RETURN(in_tensors_.at(i));
  }
  CHECK_LESS_RETURN(out_tensors_.size(), 3);
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    CHECK_NULL_RETURN(out_tensors_.at(i));
  }
  CHECK_NULL_RETURN(lstm_param_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LstmFp16CPUKernel::ReSize() {
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Lstm fp16 InitParam error.";
    return RET_ERROR;
  }

  FreeTmpBuffer();
  ret = InitInputWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Lstm fp16 InitInputWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  ret = InitStateWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Lstm fp16 InitStateWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16CPUKernel::MallocRunBuffer() {
  for (int i = 0; i < 6; i++) {
    buffer_[i] = nullptr;
  }
  buffer_[packed_input_index] = reinterpret_cast<float16_t *>(
    ms_context_->allocator->Malloc(lstm_param_->input_row_align_ * lstm_param_->input_size_ * sizeof(float16_t)));
  if (buffer_[packed_input_index] == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc input * weight left matirx error.";
    return RET_ERROR;
  }

  buffer_[input_gate_index] = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(
    gate_num * lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
  if (buffer_[input_gate_index] == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state * weight left matirx error.";
    return RET_ERROR;
  }

  if (!is_vec_) {
    buffer_[packed_state_index] = reinterpret_cast<float16_t *>(
      ms_context_->allocator->Malloc(lstm_param_->state_row_align_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
    if (buffer_[packed_state_index] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  buffer_[state_gate_index] = reinterpret_cast<float16_t *>(
    ms_context_->allocator->Malloc(gate_num * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
  if (buffer_[state_gate_index] == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state gate buffer_ error.";
    return RET_ERROR;
  }

  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    int buffer_size = lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t);
    buffer_[cell_state_index] = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(buffer_size));
    if (buffer_[cell_state_index] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state_buffer for cell error.";
      return RET_ERROR;
    }
  }
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    int buffer_size = lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t);
    buffer_[hidden_state_index] = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(buffer_size));
    if (buffer_[hidden_state_index] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state_buffer for hidden error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LstmFp16CPUKernel::Run() {
  auto input = in_tensors_.at(0);
  auto input_ptr = reinterpret_cast<float16_t *>(input->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output = out_tensors_.at(0);
  auto output_ptr = reinterpret_cast<float16_t *>(output->data());
  CHECK_NULL_RETURN(output_ptr);

  auto hidden_state = in_tensors_.at(4);
  CHECK_NULL_RETURN(hidden_state->data());
  auto cell_state = in_tensors_.at(5);
  CHECK_NULL_RETURN(cell_state->data());

  auto output_hidden_state = out_tensors_[1];
  CHECK_NULL_RETURN(output_hidden_state->data());
  memcpy(output_hidden_state->data(), hidden_state->data(), hidden_state->ElementsNum() * sizeof(float16_t));
  auto output_cell_state = out_tensors_[2];
  CHECK_NULL_RETURN(output_cell_state->data());
  memcpy(output_cell_state->data(), cell_state->data(), cell_state->ElementsNum() * sizeof(float16_t));

  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel MallocRunBuffer error.";
    FreeRunBuffer();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(weight_i_ptr_);
  CHECK_NULL_RETURN(weight_h_ptr_);
  CHECK_NULL_RETURN(input_bias_);
  CHECK_NULL_RETURN(state_bias_);
  LstmFp16(output_ptr, input_ptr, weight_i_ptr_, weight_h_ptr_, input_bias_, state_bias_,
           reinterpret_cast<float16_t *>(output_hidden_state->data()),
           reinterpret_cast<float16_t *>(output_cell_state->data()), buffer_, lstm_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LSTM, LiteKernelCreator<LstmFp16CPUKernel>)
}  // namespace mindspore::kernel
