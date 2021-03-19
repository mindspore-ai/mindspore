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

#include "src/runtime/kernel/arm/fp32/lstm_fp32.h"
#include <float.h>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LSTM;

namespace mindspore::kernel {
void LstmCPUKernel::FreeTmpBuffer() {
  if (weight_i_ptr_ != nullptr) {
    free(weight_i_ptr_);
    weight_i_ptr_ = nullptr;
  }
  if (input_bias_ != nullptr) {
    free(input_bias_);
    input_bias_ = nullptr;
  }
  if (!is_vec_) {
    if (weight_h_ptr_ != nullptr) {
      free(weight_h_ptr_);
      weight_h_ptr_ = nullptr;
    }
  }
  if (state_bias_ != nullptr) {
    free(state_bias_);
    state_bias_ = nullptr;
  }
}

void LstmCPUKernel::FreeRunBuffer() {
  context_->allocator->Free(buffer_[0]);
  context_->allocator->Free(buffer_[1]);
  if (!is_vec_) {
    context_->allocator->Free(buffer_[2]);
  }
  context_->allocator->Free(buffer_[3]);
  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    context_->allocator->Free(buffer_[4]);
  }
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    context_->allocator->Free(buffer_[5]);
  }
}

int LstmCPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  auto weight_i = in_tensors_.at(1);
  MS_ASSERT(weight_i != nullptr);
  weight_i_ptr_ = reinterpret_cast<float *>(
    malloc(weight_batch_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float)));
  if (weight_i_ptr_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc weight_i_ptr_ error.";
    return RET_ERROR;
  }
  auto weight_i_data = reinterpret_cast<float *>(weight_i->data_c());
  PackLstmWeight(weight_i_ptr_, weight_i_data, weight_batch_, lstm_param_->input_size_, lstm_param_->hidden_size_,
                 lstm_param_->input_col_align_);

  // input bias
  input_bias_ = reinterpret_cast<float *>(malloc(weight_batch_ * lstm_param_->input_col_align_ * sizeof(float)));
  if (input_bias_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc input_bias_ error.";
    return RET_ERROR;
  }
  memset(input_bias_, 0, weight_batch_ * lstm_param_->input_col_align_ * sizeof(float));
  PackLstmBias(input_bias_, reinterpret_cast<float *>(in_tensors_.at(3)->data_c()), weight_batch_,
               lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_);
  return RET_OK;
}

int LstmCPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_h = in_tensors_.at(2);
  MS_ASSERT(weight_h != nullptr);
  auto weight_h_data = reinterpret_cast<float *>(weight_h->data_c());
  if (!is_vec_) {
    weight_h_ptr_ = reinterpret_cast<float *>(
      malloc(weight_batch_ * lstm_param_->state_col_align_ * lstm_param_->hidden_size_ * sizeof(float)));
    if (weight_h_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc weight_h_ptr_ error.";
      return RET_ERROR;
    }
    PackLstmWeight(weight_h_ptr_, weight_h_data, weight_batch_, lstm_param_->hidden_size_, lstm_param_->hidden_size_,
                   lstm_param_->state_col_align_);
  } else {
    weight_h_ptr_ = weight_h_data;
  }

  // state bias
  state_bias_ = reinterpret_cast<float *>(malloc(weight_batch_ * lstm_param_->state_col_align_ * sizeof(float)));
  if (state_bias_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc state_bias_ error.";
    return RET_ERROR;
  }
  memset(state_bias_, 0, weight_batch_ * lstm_param_->state_col_align_ * sizeof(float));
  auto state_bias = reinterpret_cast<float *>(in_tensors_.at(3)->data_c()) + 4 * lstm_param_->hidden_size_;
  PackLstmBias(state_bias_, state_bias, weight_batch_, lstm_param_->hidden_size_, lstm_param_->state_col_align_,
               lstm_param_->bidirectional_);
  return RET_OK;
}

int LstmCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  lstm_param_->seq_len_ = in_shape.at(0);
  lstm_param_->batch_ = in_shape.at(1);
  lstm_param_->input_size_ = in_shape.at(2);

  auto weight_i = in_tensors_.at(1);
  MS_ASSERT(weight_i != nullptr);
  std::vector<int> w_shape = weight_i->shape();
  lstm_param_->hidden_size_ = w_shape.at(1) / 4;

  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? 2 * lstm_param_->batch_ * lstm_param_->hidden_size_
                                                          : lstm_param_->batch_ * lstm_param_->hidden_size_;
  weight_batch_ = lstm_param_->bidirectional_ ? 8 : 4;

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
  lstm_param_->input_row_align_ = UP_ROUND(lstm_param_->seq_len_ * lstm_param_->batch_, row_tile_);
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, col_tile_);

  is_vec_ = lstm_param_->batch_ == 1;
  lstm_param_->state_row_align_ = is_vec_ ? 1 : UP_ROUND(lstm_param_->batch_, row_tile_);
  lstm_param_->state_col_align_ = is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, col_tile_);
  return RET_OK;
}

int LstmCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LstmCPUKernel::ReSize() {
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitParam error.";
    return RET_ERROR;
  }

  FreeTmpBuffer();
  ret = InitInputWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitInputWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  ret = InitStateWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitStateWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmCPUKernel::MallocRunBuffer() {
  for (int i = 0; i < 6; i++) {
    buffer_[i] = nullptr;
  }
  buffer_[0] = reinterpret_cast<float *>(
    context_->allocator->Malloc(lstm_param_->input_row_align_ * lstm_param_->input_size_ * sizeof(float)));
  if (buffer_[0] == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc input * weight left matirx error.";
    return RET_ERROR;
  }

  buffer_[1] = reinterpret_cast<float *>(context_->allocator->Malloc(4 * lstm_param_->seq_len_ * lstm_param_->batch_ *
                                                                     lstm_param_->hidden_size_ * sizeof(float)));
  if (buffer_[1] == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc input * weight result matirx error.";
    return RET_ERROR;
  }

  if (!is_vec_) {
    buffer_[2] = reinterpret_cast<float *>(
      context_->allocator->Malloc(lstm_param_->state_row_align_ * lstm_param_->hidden_size_ * sizeof(float)));
    if (buffer_[2] == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  buffer_[3] = reinterpret_cast<float *>(
    context_->allocator->Malloc(4 * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float)));
  if (buffer_[3] == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc state gate buffer error.";
    return RET_ERROR;
  }

  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    auto buffer_size = lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float);
    buffer_[4] = reinterpret_cast<float *>(context_->allocator->Malloc(buffer_size));
    if (buffer_[4] == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc state_buffer for cell error.";
      return RET_ERROR;
    }
  }
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    auto buffer_size = lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float);
    buffer_[5] = reinterpret_cast<float *>(context_->allocator->Malloc(buffer_size));
    if (buffer_[5] == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc state_buffer for hidden error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LstmCPUKernel::Run() {
  auto input = in_tensors_.at(kInputIndex);
  MS_ASSERT(input != nullptr);
  auto hidden_state = in_tensors_.at(4);
  MS_ASSERT(hidden_state != nullptr);
  auto cell_state = in_tensors_.at(5);
  MS_ASSERT(cell_state != nullptr);
  auto output = out_tensors_.at(0);
  MS_ASSERT(output != nullptr);

  auto input_ptr = reinterpret_cast<float *>(input->data_c());
  MS_ASSERT(input_ptr);
  auto output_ptr = reinterpret_cast<float *>(output->data_c());
  MS_ASSERT(output_ptr);
  auto output_hidden_state = out_tensors_[1];
  memcpy(output_hidden_state->data_c(), hidden_state->data_c(), hidden_state->ElementsNum() * sizeof(float));
  auto output_cell_state = out_tensors_[2];
  memcpy(output_cell_state->data_c(), cell_state->data_c(), cell_state->ElementsNum() * sizeof(float));

  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel MallocRunBuffer error.";
    return RET_ERROR;
  }

  MS_ASSERT(weight_h_ptr_);
  MS_ASSERT(weight_i_ptr_);
  MS_ASSERT(input_bias_);
  MS_ASSERT(state_bias_);
  Lstm(output_ptr, input_ptr, weight_i_ptr_, weight_h_ptr_, input_bias_, state_bias_,
       reinterpret_cast<float *>(output_hidden_state->data_c()), reinterpret_cast<float *>(output_cell_state->data_c()),
       buffer_, lstm_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LSTM, LiteKernelCreator<LstmCPUKernel>)
}  // namespace mindspore::kernel
