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
  if (!is_vec_) {
    if (weight_g_ptr_ != nullptr) {
      free(weight_g_ptr_);
      weight_g_ptr_ = nullptr;
    }
    if (weight_r_ptr_ != nullptr) {
      free(weight_r_ptr_);
      weight_r_ptr_ = nullptr;
    }
    if (bias_ptr_ != nullptr) {
      free(bias_ptr_);
      bias_ptr_ = nullptr;
    }
  }
}

void GruCPUKernel::FreeRunBuffer() {
  context_->allocator->Free(gate_buffer_);
  if (!is_vec_) {
    for (int i = 0; i < 2; i++) {
      context_->allocator->Free(matmul_buffer_[i]);
    }
  }
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

  gru_param_->input_step_ = gru_param_->batch_ * gru_param_->input_size_;
  gru_param_->output_step_ = gru_param_->bidirectional_ ? 2 * gru_param_->batch_ * gru_param_->hidden_size_
                                                        : gru_param_->batch_ * gru_param_->hidden_size_;

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
  is_vec_ = gru_param_->batch_ == 1;
  gru_param_->row_align_ = is_vec_ ? 1 : UP_ROUND(gru_param_->batch_, row_tile_);
  gru_param_->col_align_ = is_vec_ ? gru_param_->hidden_size_ : UP_ROUND(gru_param_->hidden_size_, col_tile_);
  return RET_OK;
}

int GruCPUKernel::InitWeightBias() {
  auto weight_batch = gru_param_->bidirectional_ ? 6 : 3;
  if (!is_vec_) {
    // malloc and init input * weight right matrix buffer
    auto weight_g = in_tensors_.at(1);
    MS_ASSERT(weight_g != nullptr);
    weight_g_ptr_ = reinterpret_cast<float *>(
      malloc(weight_batch * gru_param_->col_align_ * gru_param_->input_size_ * sizeof(float)));
    if (weight_g_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc weight_g_ptr_ error.";
      return RET_ERROR;
    }
    auto weight_i_data = reinterpret_cast<float *>(weight_g->data_c());
    PackLstmWeight(weight_g_ptr_, weight_i_data, weight_batch, gru_param_->input_size_, gru_param_->hidden_size_,
                   gru_param_->col_align_);

    // malloc and init state * weight right matrix buffer
    auto weight_r = in_tensors_.at(2);
    MS_ASSERT(weight_r != nullptr);
    weight_r_ptr_ = reinterpret_cast<float *>(
      malloc(weight_batch * gru_param_->col_align_ * gru_param_->hidden_size_ * sizeof(float)));
    if (weight_r_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc weight_r_ptr_ error.";
      return RET_ERROR;
    }
    auto weight_r_data = reinterpret_cast<float *>(weight_r->data_c());
    PackLstmWeight(weight_r_ptr_, weight_r_data, weight_batch, gru_param_->hidden_size_, gru_param_->hidden_size_,
                   gru_param_->col_align_);

    // init bias
    int bias_batch = gru_param_->bidirectional_ ? 16 : 8;
    bias_ptr_ = reinterpret_cast<float *>(malloc(bias_batch * gru_param_->col_align_ * sizeof(float)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc bias_ptr_ error.";
      return RET_ERROR;
    }
    memset(bias_ptr_, 0, bias_batch * gru_param_->col_align_ * sizeof(float));
    auto bias_data = reinterpret_cast<float *>(in_tensors_.at(3)->data_c());
    for (int i = 0; i < bias_batch; i++) {
      auto src_batch = bias_data + i * gru_param_->hidden_size_;
      auto dst_batch = bias_ptr_ + i * gru_param_->col_align_;
      memcpy(dst_batch, src_batch, gru_param_->hidden_size_ * sizeof(float));
    }
  }
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
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel InitWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int GruCPUKernel::MallocRunBuffer() {
  if (!is_vec_) {
    matmul_buffer_[0] = reinterpret_cast<float *>(
      context_->allocator->Malloc(3 * gru_param_->row_align_ * gru_param_->input_size_ * sizeof(float)));
    if (matmul_buffer_[0] == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc input * weight left matirx error.";
      return RET_ERROR;
    }

    matmul_buffer_[1] = reinterpret_cast<float *>(
      context_->allocator->Malloc(3 * gru_param_->row_align_ * gru_param_->hidden_size_ * sizeof(float)));
    if (matmul_buffer_[1] == nullptr) {
      MS_LOG(ERROR) << "GruCPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }
  gate_buffer_ = reinterpret_cast<float *>(
    context_->allocator->Malloc(6 * gru_param_->batch_ * gru_param_->hidden_size_ * sizeof(float)));
  if (gate_buffer_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc gate_buffer error.";
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

  if (is_vec_) {
    weight_g_ptr_ = reinterpret_cast<float *>(in_tensors_[1]->data_c());
    weight_r_ptr_ = reinterpret_cast<float *>(in_tensors_[2]->data_c());
    bias_ptr_ = reinterpret_cast<float *>(in_tensors_[3]->data_c());
  }
  MS_ASSERT(weight_g_ptr_ != nullptr);
  MS_ASSERT(weight_r_ptr_ != nullptr);
  MS_ASSERT(bias_ptr_ != nullptr);
  MS_ASSERT(gate_buffer_ != nullptr);
  Gru(output_ptr, input_ptr, weight_g_ptr_, weight_r_ptr_, bias_ptr_,
      reinterpret_cast<float *>(output_hidden_state->data_c()), gate_buffer_, matmul_buffer_, check_seq_len,
      gru_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GRU, LiteKernelCreator<GruCPUKernel>)
}  // namespace mindspore::kernel
