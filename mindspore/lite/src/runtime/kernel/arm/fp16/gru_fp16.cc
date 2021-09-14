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
#include "src/runtime/kernel/arm/fp16/gru_fp16.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/gru_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/lstm_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GRU;

namespace mindspore::kernel {
void GruFp16CPUKernel::FreeTmpBuffer() {
  if (weight_g_ptr_ != nullptr) {
    free(weight_g_ptr_);
    weight_g_ptr_ = nullptr;
  }
  if (input_bias_ != nullptr) {
    free(input_bias_);
    input_bias_ = nullptr;
  }
  if (weight_r_ptr_ != nullptr) {
    free(weight_r_ptr_);
    weight_r_ptr_ = nullptr;
  }
  if (state_bias_ != nullptr) {
    free(state_bias_);
    state_bias_ = nullptr;
  }
}

void GruFp16CPUKernel::FreeRunBuffer() {
  ms_context_->allocator->Free(buffer_[packed_input_index]);
  ms_context_->allocator->Free(buffer_[input_gate_index]);
  if (!is_vec_) {
    ms_context_->allocator->Free(buffer_[packed_state_index]);
  }
  ms_context_->allocator->Free(buffer_[state_gate_index]);
}

int GruFp16CPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  gru_param_->seq_len_ = in_shape.at(0);
  gru_param_->batch_ = in_shape.at(1);
  gru_param_->input_size_ = in_shape.at(2);

  auto weight_g = in_tensors_.at(1);
  MS_ASSERT(weight_g != nullptr);
  std::vector<int> w_shape = weight_g->shape();
  gru_param_->hidden_size_ = w_shape.at(1) / gate_num;
  weight_batch_ = gru_param_->bidirectional_ ? 2 * gate_num : gate_num;
  gru_param_->output_step_ = gru_param_->bidirectional_ ? 2 * gru_param_->batch_ * gru_param_->hidden_size_
                                                        : gru_param_->batch_ * gru_param_->hidden_size_;

  gru_param_->input_row_align_ = UP_ROUND(gru_param_->seq_len_ * gru_param_->batch_, C16NUM);
  gru_param_->input_col_align_ = UP_ROUND(gru_param_->hidden_size_, C8NUM);

  is_vec_ = gru_param_->batch_ == 1;
  gru_param_->state_row_align_ = is_vec_ ? gru_param_->batch_ : UP_ROUND(gru_param_->batch_, C16NUM);
  gru_param_->state_col_align_ = is_vec_ ? gru_param_->hidden_size_ : UP_ROUND(gru_param_->hidden_size_, C8NUM);
  return RET_OK;
}

int GruFp16CPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  auto weight_g = in_tensors_.at(1);
  CHECK_NULL_RETURN(weight_g->data());
  weight_g_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_batch_ * gru_param_->input_col_align_ * gru_param_->input_size_ * sizeof(float16_t)));
  if (weight_g_ptr_ == nullptr) {
    MS_LOG(ERROR) << "GruFp16CPUKernel malloc weight_g_ptr_ error.";
    return RET_ERROR;
  }
  if (weight_g->data_type() == kNumberTypeFloat32) {
    PackLstmWeightFp32ToFp16(weight_g_ptr_, reinterpret_cast<float *>(weight_g->data()), weight_batch_,
                             gru_param_->input_size_, gru_param_->hidden_size_, gru_param_->input_col_align_);
  } else if (weight_g->data_type() == kNumberTypeFloat16) {
    PackLstmWeightFp16(weight_g_ptr_, reinterpret_cast<float16_t *>(weight_g->data()), weight_batch_,
                       gru_param_->input_size_, gru_param_->hidden_size_, gru_param_->input_col_align_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight_g tensor for gru.";
    return RET_ERROR;
  }

  // input bias
  auto bias = in_tensors_.at(3);
  CHECK_NULL_RETURN(bias->data());
  input_bias_ = reinterpret_cast<float16_t *>(malloc(weight_batch_ * gru_param_->input_col_align_ * sizeof(float16_t)));
  if (input_bias_ == nullptr) {
    MS_LOG(ERROR) << "GruFp16CPUKernel malloc input_bias_ error.";
    return RET_ERROR;
  }
  memset(input_bias_, 0, weight_batch_ * gru_param_->input_col_align_ * sizeof(float16_t));
  if (bias->data_type() == kNumberTypeFloat32) {
    PackLstmBiasFp32ToFp16(input_bias_, reinterpret_cast<float *>(bias->data()), weight_batch_,
                           gru_param_->hidden_size_, gru_param_->input_col_align_, gru_param_->bidirectional_);
  } else if (bias->data_type() == kNumberTypeFloat16) {
    PackLstmBiasFp16(input_bias_, reinterpret_cast<float16_t *>(bias->data()), weight_batch_, gru_param_->hidden_size_,
                     gru_param_->input_col_align_, gru_param_->bidirectional_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for gru.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_r = in_tensors_.at(2);
  CHECK_NULL_RETURN(weight_r->data());
  weight_r_ptr_ = reinterpret_cast<float16_t *>(
    malloc(weight_batch_ * gru_param_->state_col_align_ * gru_param_->hidden_size_ * sizeof(float16_t)));
  if (weight_r_ptr_ == nullptr) {
    MS_LOG(ERROR) << "GruFp16CPUKernel malloc weight_r_ptr_ error.";
    return RET_ERROR;
  }

  if (!is_vec_) {
    if (weight_r->data_type() == kNumberTypeFloat32) {
      PackLstmWeightFp32ToFp16(weight_r_ptr_, reinterpret_cast<float *>(weight_r->data()), weight_batch_,
                               gru_param_->hidden_size_, gru_param_->hidden_size_, gru_param_->state_col_align_);
    } else if (weight_r->data_type() == kNumberTypeFloat16) {
      PackLstmWeightFp16(weight_r_ptr_, reinterpret_cast<float16_t *>(weight_r->data()), weight_batch_,
                         gru_param_->hidden_size_, gru_param_->hidden_size_, gru_param_->state_col_align_);
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_r tensor for gru.";
      return RET_ERROR;
    }
  } else {
    if (weight_r->data_type() == kNumberTypeFloat32) {
      Float32ToFloat16(reinterpret_cast<float *>(weight_r->data()), weight_r_ptr_, weight_r->ElementsNum());
    } else if (weight_r->data_type() == kNumberTypeFloat16) {
      memcpy(weight_r_ptr_, reinterpret_cast<float16_t *>(weight_r->data()), weight_r->Size());
    } else {
      MS_LOG(ERROR) << "Unsupported data type of weight_r tensor for gru.";
      return RET_ERROR;
    }
  }

  // state bias
  auto bias = in_tensors_.at(3);
  CHECK_NULL_RETURN(bias->data());
  state_bias_ = reinterpret_cast<float16_t *>(malloc(weight_batch_ * gru_param_->state_col_align_ * sizeof(float16_t)));
  if (state_bias_ == nullptr) {
    MS_LOG(ERROR) << "GruFp16CPUKernel malloc state_bias_ error.";
    return RET_ERROR;
  }
  memset(state_bias_, 0, weight_batch_ * gru_param_->state_col_align_ * sizeof(float16_t));
  if (bias->data_type() == kNumberTypeFloat32) {
    auto state_bias_data = reinterpret_cast<float *>(bias->data()) + gate_num * gru_param_->hidden_size_;
    PackLstmBiasFp32ToFp16(state_bias_, state_bias_data, weight_batch_, gru_param_->hidden_size_,
                           gru_param_->state_col_align_, gru_param_->bidirectional_);
  } else if (bias->data_type() == kNumberTypeFloat16) {
    auto state_bias_data = reinterpret_cast<float16_t *>(bias->data()) + gate_num * gru_param_->hidden_size_;
    PackLstmBiasFp16(state_bias_, state_bias_data, weight_batch_, gru_param_->hidden_size_,
                     gru_param_->state_col_align_, gru_param_->bidirectional_);
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for gru.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 5);
  CHECK_LESS_RETURN(out_tensors_.size(), 2);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GruFp16CPUKernel::ReSize() {
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruFp16CPUKernel InitParam error.";
    return RET_ERROR;
  }

  FreeTmpBuffer();
  ret = InitInputWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruFp16CPUKernel InitInputWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  ret = InitStateWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruFp16CPUKernel InitStateWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::MallocRunBuffer() {
  for (int i = 0; i < 4; i++) {
    buffer_[i] = nullptr;
  }
  buffer_[packed_input_index] = reinterpret_cast<float16_t *>(
    ms_context_->allocator->Malloc(gru_param_->input_row_align_ * gru_param_->input_size_ * sizeof(float16_t)));
  if (buffer_[packed_input_index] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input * weight left matirx error.";
    return RET_ERROR;
  }

  buffer_[input_gate_index] = reinterpret_cast<float16_t *>(ms_context_->allocator->Malloc(
    gate_num * gru_param_->seq_len_ * gru_param_->batch_ * gru_param_->hidden_size_ * sizeof(float16_t)));
  if (buffer_[input_gate_index] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input * weight result matirx error.";
    return RET_ERROR;
  }

  if (!is_vec_) {
    buffer_[packed_state_index] = reinterpret_cast<float16_t *>(
      ms_context_->allocator->Malloc(gru_param_->state_row_align_ * gru_param_->hidden_size_ * sizeof(float16_t)));
    if (buffer_[packed_state_index] == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  buffer_[state_gate_index] = reinterpret_cast<float16_t *>(
    ms_context_->allocator->Malloc(gate_num * gru_param_->batch_ * gru_param_->hidden_size_ * sizeof(float16_t)));
  if (buffer_[state_gate_index] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc state gate buffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::Run() {
  auto input = in_tensors_.at(kInputIndex);
  auto input_ptr = reinterpret_cast<float16_t *>(input->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output = out_tensors_.at(0);
  auto output_ptr = reinterpret_cast<float16_t *>(output->data());
  CHECK_NULL_RETURN(output_ptr);
  auto hidden_state = in_tensors_.at(4);
  auto output_hidden_state = out_tensors_.at(1);
  CHECK_NULL_RETURN(output_hidden_state->data());
  CHECK_NULL_RETURN(hidden_state->data());
  memcpy(output_hidden_state->data(), hidden_state->data(), hidden_state->ElementsNum() * sizeof(float16_t));
  int check_seq_len = gru_param_->seq_len_;
  if (in_tensors_.size() == 6) {
    MS_ASSERT(in_tensors_.at(5) != nullptr);
    int *seq_len = reinterpret_cast<int *>(in_tensors_.at(5)->data());
    MS_ASSERT(seq_len != nullptr);
    if (!std::equal(seq_len + 1, seq_len + gru_param_->batch_, seq_len)) {
      MS_LOG(ERROR) << "different batch seq_len is currently not supported";
      return RET_ERROR;
    }
    check_seq_len = MSMIN(check_seq_len, MSMAX(0, seq_len[0]));
  }

  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruFp16CPUKernel MallocRunBuffer error.";
    FreeRunBuffer();
    return RET_ERROR;
  }
  MS_ASSERT(weight_g_ptr_ != nullptr);
  MS_ASSERT(weight_r_ptr_ != nullptr);
  MS_ASSERT(input_bias_ != nullptr);
  MS_ASSERT(state_bias_ != nullptr);
  GruFp16(output_ptr, input_ptr, weight_g_ptr_, weight_r_ptr_, input_bias_, state_bias_,
          reinterpret_cast<float16_t *>(output_hidden_state->data()), buffer_, check_seq_len, gru_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_GRU, LiteKernelCreator<GruFp16CPUKernel>)
}  // namespace mindspore::kernel
