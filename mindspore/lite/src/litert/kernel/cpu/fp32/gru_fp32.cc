/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/gru_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/gru_fp32.h"
#include "nnacl/fp32/lstm_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GRU;

namespace mindspore::kernel {
void GruCPUKernel::FreeTmpBuffer() {
  if (weight_g_ptr_ != nullptr) {
    free(weight_g_ptr_);
    weight_g_ptr_ = nullptr;
  }
  if (input_bias_ != nullptr) {
    free(input_bias_);
    input_bias_ = nullptr;
  }
  if (!is_vec_) {
    if (weight_r_ptr_ != nullptr) {
      free(weight_r_ptr_);
      weight_r_ptr_ = nullptr;
    }
  }
  if (state_bias_ != nullptr) {
    free(state_bias_);
    state_bias_ = nullptr;
  }
}

void GruCPUKernel::FreeRunBuffer() {
  ms_context_->allocator->Free(buffer_[packed_input_index]);
  ms_context_->allocator->Free(buffer_[input_gate_index]);
  if (!is_vec_) {
    ms_context_->allocator->Free(buffer_[packed_state_index]);
  }
  ms_context_->allocator->Free(buffer_[state_gate_index]);
}

int GruCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  gru_param_->seq_len_ = in_shape.at(0);
  gru_param_->batch_ = in_shape.at(1);
  gru_param_->input_size_ = in_shape.at(2);

  auto weight_g = in_tensors_.at(weight_g_index);
  MS_ASSERT(weight_g != nullptr);
  std::vector<int> w_shape = weight_g->shape();
  gru_param_->hidden_size_ = w_shape.at(1) / gate_num;

  MS_CHECK_INT_MUL_NOT_OVERFLOW(gru_param_->batch_, gru_param_->hidden_size_, RET_ERROR);
  int gru_bh = gru_param_->batch_ * gru_param_->hidden_size_;
  if (gru_param_->bidirectional_) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(C2NUM, gru_bh, RET_ERROR);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(C2NUM, gate_num, RET_ERROR);
    gru_param_->output_step_ = C2NUM * gru_bh;
    weight_batch_ = C2NUM * gate_num;
  } else {
    gru_param_->output_step_ = gru_bh;
    weight_batch_ = gate_num;
  }

#ifdef ENABLE_AVX
  row_tile_ = C6NUM;
  col_tile_ = C16NUM;
#elif defined(ENABLE_ARM32)
  row_tile_ = C12NUM;
  col_tile_ = C4NUM;
#elif defined(ENABLE_SSE)
  row_tile_ = C4NUM;
  col_tile_ = C8NUM;
#else
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  MS_CHECK_INT_MUL_NOT_OVERFLOW(gru_param_->seq_len_, gru_param_->batch_, RET_ERROR);
  gru_param_->input_row_align_ = UP_ROUND(gru_param_->seq_len_ * gru_param_->batch_, row_tile_);
  gru_param_->input_col_align_ = UP_ROUND(gru_param_->hidden_size_, col_tile_);

  is_vec_ = gru_param_->batch_ == 1;
  gru_param_->state_row_align_ = is_vec_ ? 1 : UP_ROUND(gru_param_->batch_, row_tile_);
  gru_param_->state_col_align_ = is_vec_ ? gru_param_->hidden_size_ : UP_ROUND(gru_param_->hidden_size_, col_tile_);
  return RET_OK;
}

int GruCPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  auto weight_g = in_tensors_.at(weight_g_index);
  MS_ASSERT(weight_g != nullptr);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_batch_, gru_param_->input_col_align_, RET_ERROR);
  int weight_size = weight_batch_ * gru_param_->input_col_align_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_size, gru_param_->input_size_, RET_ERROR);
  weight_g_ptr_ = reinterpret_cast<float *>(malloc(weight_size * gru_param_->input_size_ * sizeof(float)));
  if (weight_g_ptr_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc weight_g_ptr_ error.";
    return RET_ERROR;
  }
  auto weight_g_data = reinterpret_cast<float *>(weight_g->data());
  CHECK_NULL_RETURN(weight_g_data);
  PackLstmWeight(weight_g_ptr_, weight_g_data, weight_batch_, gru_param_->input_size_, gru_param_->hidden_size_,
                 gru_param_->input_col_align_, nullptr);

  // input bias
  input_bias_ = reinterpret_cast<float *>(malloc(weight_size * sizeof(float)));
  if (input_bias_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input_bias_ error.";
    return RET_ERROR;
  }
  memset(input_bias_, 0, weight_size * sizeof(float));
  auto bias_g_data = reinterpret_cast<float *>(in_tensors_.at(bias_index)->data());
  CHECK_NULL_RETURN(bias_g_data);
  PackLstmBias(input_bias_, bias_g_data, weight_batch_, gru_param_->hidden_size_, gru_param_->input_col_align_,
               gru_param_->bidirectional_, nullptr);
  return RET_OK;
}

int GruCPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_r = in_tensors_.at(weight_r_index);
  MS_ASSERT(weight_r != nullptr);
  auto weight_r_data = reinterpret_cast<float *>(weight_r->data());
  CHECK_NULL_RETURN(weight_r_data);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_batch_, gru_param_->state_col_align_, RET_ERROR);
  int weight_plane_size = weight_batch_ * gru_param_->state_col_align_;
  if (!is_vec_) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_plane_size, gru_param_->hidden_size_, RET_ERROR);
    weight_r_ptr_ = reinterpret_cast<float *>(malloc(weight_plane_size * gru_param_->hidden_size_ * sizeof(float)));
    if (weight_r_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc weight_r_ptr_ error.";
      return RET_ERROR;
    }
    PackLstmWeight(weight_r_ptr_, weight_r_data, weight_batch_, gru_param_->hidden_size_, gru_param_->hidden_size_,
                   gru_param_->state_col_align_, nullptr);
  } else {
    weight_r_ptr_ = weight_r_data;
  }

  // state bias
  state_bias_ = reinterpret_cast<float *>(malloc(weight_plane_size * sizeof(float)));
  if (state_bias_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc state_bias_ error.";
    return RET_ERROR;
  }
  memset(state_bias_, 0, weight_plane_size * sizeof(float));
  auto bias_r_data = reinterpret_cast<float *>(in_tensors_.at(bias_index)->data());
  CHECK_NULL_RETURN(bias_r_data);

  auto state_bias = bias_r_data + gate_num * gru_param_->hidden_size_;
  PackLstmBias(state_bias_, state_bias, weight_batch_, gru_param_->hidden_size_, gru_param_->state_col_align_,
               gru_param_->bidirectional_, nullptr);
  return RET_OK;
}

int GruCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_5D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GruCPUKernel::ReSize() {
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel InitParam error.";
    return RET_ERROR;
  }

  FreeTmpBuffer();
  ret = InitInputWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel InitInputWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  ret = InitStateWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel InitStateWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int GruCPUKernel::MallocRunBuffer() {
  for (int i = 0; i < 4; i++) {
    buffer_[i] = nullptr;
  }
  MS_CHECK_INT_MUL_NOT_OVERFLOW(gru_param_->input_row_align_, gru_param_->input_size_, RET_ERROR);
  buffer_[packed_input_index] = reinterpret_cast<float *>(
    ms_context_->allocator->Malloc(gru_param_->input_row_align_ * gru_param_->input_size_ * sizeof(float)));
  if (buffer_[packed_input_index] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input * weight left matirx error.";
    return RET_ERROR;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(gate_num * gru_param_->hidden_size_, gru_param_->batch_, RET_ERROR);
  int tmp_size = gate_num * gru_param_->hidden_size_ * gru_param_->batch_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(tmp_size, gru_param_->seq_len_, RET_ERROR);
  buffer_[input_gate_index] =
    reinterpret_cast<float *>(ms_context_->allocator->Malloc(gru_param_->seq_len_ * tmp_size * sizeof(float)));
  if (buffer_[input_gate_index] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input * weight result matirx error.";
    return RET_ERROR;
  }

  if (!is_vec_) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(gru_param_->state_row_align_, gru_param_->hidden_size_, RET_ERROR);
    buffer_[packed_state_index] = reinterpret_cast<float *>(
      ms_context_->allocator->Malloc(gru_param_->state_row_align_ * gru_param_->hidden_size_ * sizeof(float)));
    if (buffer_[packed_state_index] == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  buffer_[state_gate_index] = reinterpret_cast<float *>(ms_context_->allocator->Malloc(tmp_size * sizeof(float)));
  if (buffer_[state_gate_index] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc state gate buffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruCPUKernel::Run() {
  auto input = in_tensors_.at(kInputIndex);
  auto input_ptr = reinterpret_cast<float *>(input->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output = out_tensors_.at(0);
  auto output_ptr = reinterpret_cast<float *>(output->data());
  CHECK_NULL_RETURN(output_ptr);
  auto hidden_state = in_tensors_.at(4);
  auto output_hidden_state = out_tensors_.at(1);
  CHECK_NULL_RETURN(output_hidden_state->data());
  CHECK_NULL_RETURN(hidden_state->data());
  (void)memcpy(output_hidden_state->data(), hidden_state->data(), hidden_state->ElementsNum() * sizeof(float));
  int check_seq_len = gru_param_->seq_len_;

  if (in_tensors_.size() == 6) {
    auto seq_len = reinterpret_cast<int *>(in_tensors_.at(DIMENSION_5D)->data());
    CHECK_NULL_RETURN(seq_len);
    if (!std::equal(seq_len + 1, seq_len + gru_param_->batch_, seq_len)) {
      MS_LOG(ERROR) << "different batch seq_len is currently not supported";
      return RET_ERROR;
    }
    check_seq_len = MSMIN(check_seq_len, MSMAX(0, seq_len[0]));
  }
  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel MallocRunBuffer error.";
    FreeRunBuffer();
    return RET_ERROR;
  }

  MS_ASSERT(weight_g_ptr_ != nullptr);
  MS_ASSERT(weight_r_ptr_ != nullptr);
  MS_ASSERT(input_bias_ != nullptr);
  MS_ASSERT(state_bias_ != nullptr);
  Gru(output_ptr, input_ptr, weight_g_ptr_, weight_r_ptr_, input_bias_, state_bias_,
      reinterpret_cast<float *>(output_hidden_state->data()), buffer_, check_seq_len, gru_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GRU, LiteKernelCreator<GruCPUKernel>)
}  // namespace mindspore::kernel
