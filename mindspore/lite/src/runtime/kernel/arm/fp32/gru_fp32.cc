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
#include "src/runtime/kernel/arm/fp32/gru_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/gru_fp32.h"
#include "nnacl/fp32/lstm_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
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
  context_->allocator->Free(buffer_[0]);
  context_->allocator->Free(buffer_[1]);
  if (!is_vec_) {
    context_->allocator->Free(buffer_[2]);
  }
  context_->allocator->Free(buffer_[3]);
}

int GruCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  gru_param_->seq_len_ = in_shape.at(0);
  gru_param_->batch_ = in_shape.at(1);
  gru_param_->input_size_ = in_shape.at(2);

  auto weight_g = in_tensors_.at(1);
  MS_ASSERT(weight_g != nullptr);
  std::vector<int> w_shape = weight_g->shape();
  gru_param_->hidden_size_ = w_shape.at(1) / 3;

  gru_param_->output_step_ = gru_param_->bidirectional_ ? 2 * gru_param_->batch_ * gru_param_->hidden_size_
                                                        : gru_param_->batch_ * gru_param_->hidden_size_;
  weight_batch_ = gru_param_->bidirectional_ ? 6 : 3;

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
  auto weight_g = in_tensors_.at(1);
  MS_ASSERT(weight_g != nullptr);
  weight_g_ptr_ = reinterpret_cast<float *>(
    malloc(weight_batch_ * gru_param_->input_col_align_ * gru_param_->input_size_ * sizeof(float)));
  if (weight_g_ptr_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc weight_g_ptr_ error.";
    return RET_ERROR;
  }
  auto weight_g_data = reinterpret_cast<float *>(weight_g->data_c());
  PackLstmWeight(weight_g_ptr_, weight_g_data, weight_batch_, gru_param_->input_size_, gru_param_->hidden_size_,
                 gru_param_->input_col_align_);

  // input bias
  input_bias_ = reinterpret_cast<float *>(malloc(weight_batch_ * gru_param_->input_col_align_ * sizeof(float)));
  if (input_bias_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input_bias_ error.";
    return RET_ERROR;
  }
  memset(input_bias_, 0, weight_batch_ * gru_param_->input_col_align_ * sizeof(float));
  PackLstmBias(input_bias_, reinterpret_cast<float *>(in_tensors_.at(3)->data_c()), weight_batch_,
               gru_param_->hidden_size_, gru_param_->input_col_align_, gru_param_->bidirectional_);
  return RET_OK;
}

int GruCPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_r = in_tensors_.at(2);
  MS_ASSERT(weight_r != nullptr);
  auto weight_r_data = reinterpret_cast<float *>(weight_r->data_c());
  if (!is_vec_) {
    weight_r_ptr_ = reinterpret_cast<float *>(
      malloc(weight_batch_ * gru_param_->state_col_align_ * gru_param_->hidden_size_ * sizeof(float)));
    if (weight_r_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc weight_r_ptr_ error.";
      return RET_ERROR;
    }
    PackLstmWeight(weight_r_ptr_, weight_r_data, weight_batch_, gru_param_->hidden_size_, gru_param_->hidden_size_,
                   gru_param_->state_col_align_);
  } else {
    weight_r_ptr_ = weight_r_data;
  }

  // state bias
  state_bias_ = reinterpret_cast<float *>(malloc(weight_batch_ * gru_param_->state_col_align_ * sizeof(float)));
  if (state_bias_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc state_bias_ error.";
    return RET_ERROR;
  }
  memset(state_bias_, 0, weight_batch_ * gru_param_->state_col_align_ * sizeof(float));
  auto state_bias = reinterpret_cast<float *>(in_tensors_.at(3)->data_c()) + 3 * gru_param_->hidden_size_;
  PackLstmBias(state_bias_, state_bias, weight_batch_, gru_param_->hidden_size_, gru_param_->state_col_align_,
               gru_param_->bidirectional_);
  return RET_OK;
}

int GruCPUKernel::Init() {
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
  buffer_[0] = reinterpret_cast<float *>(
    context_->allocator->Malloc(gru_param_->input_row_align_ * gru_param_->input_size_ * sizeof(float)));
  if (buffer_[0] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input * weight left matirx error.";
    return RET_ERROR;
  }

  buffer_[1] = reinterpret_cast<float *>(context_->allocator->Malloc(3 * gru_param_->seq_len_ * gru_param_->batch_ *
                                                                     gru_param_->hidden_size_ * sizeof(float)));
  if (buffer_[1] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc input * weight result matirx error.";
    return RET_ERROR;
  }

  if (!is_vec_) {
    buffer_[2] = reinterpret_cast<float *>(
      context_->allocator->Malloc(gru_param_->state_row_align_ * gru_param_->hidden_size_ * sizeof(float)));
    if (buffer_[2] == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  buffer_[3] = reinterpret_cast<float *>(
    context_->allocator->Malloc(3 * gru_param_->batch_ * gru_param_->hidden_size_ * sizeof(float)));
  if (buffer_[3] == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc state gate buffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruCPUKernel::Run() {
  auto input = in_tensors_.at(kInputIndex);
  MS_ASSERT(input != nullptr);
  auto hidden_state = in_tensors_.at(4);
  MS_ASSERT(hidden_state != nullptr);
  auto output = out_tensors_.at(0);
  MS_ASSERT(output != nullptr);
  auto input_ptr = reinterpret_cast<float *>(input->data_c());
  MS_ASSERT(input_ptr);
  auto output_ptr = reinterpret_cast<float *>(output->data_c());
  MS_ASSERT(output_ptr);
  auto output_hidden_state = out_tensors_[1];
  memcpy(output_hidden_state->data_c(), hidden_state->data_c(), hidden_state->ElementsNum() * sizeof(float));
  int check_seq_len = gru_param_->seq_len_;

  if (in_tensors_.size() == 6) {
    auto seq_len = reinterpret_cast<int *>(in_tensors_.at(5)->data_c());
    if (!std::equal(seq_len + 1, seq_len + gru_param_->batch_, seq_len)) {
      MS_LOG(ERROR) << "different batch seq_len is currently not supported";
      return RET_ERROR;
    }
    check_seq_len = MSMIN(check_seq_len, MSMAX(0, seq_len[0]));
  }
  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel MallocRunBuffer error.";
    return RET_ERROR;
  }

  MS_ASSERT(weight_g_ptr_ != nullptr);
  MS_ASSERT(weight_r_ptr_ != nullptr);
  MS_ASSERT(input_bias_ != nullptr);
  MS_ASSERT(state_bias_ != nullptr);
  Gru(output_ptr, input_ptr, weight_g_ptr_, weight_r_ptr_, input_bias_, state_bias_,
      reinterpret_cast<float *>(output_hidden_state->data_c()), buffer_, check_seq_len, gru_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GRU, LiteKernelCreator<GruCPUKernel>)
}  // namespace mindspore::kernel
