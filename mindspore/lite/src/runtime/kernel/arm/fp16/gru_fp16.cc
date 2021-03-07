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

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GRU;

namespace mindspore::kernel {
void GruFp16CPUKernel::FreeTmpBuffer() {
  if (!is_vec_ || in_tensors_[1]->data_type() == kNumberTypeFloat32) {
    if (weight_g_ptr_ != nullptr) {
      free(weight_g_ptr_);
      weight_g_ptr_ = nullptr;
    }
  }
  if (!is_vec_ || in_tensors_[2]->data_type() == kNumberTypeFloat32) {
    if (weight_r_ptr_ != nullptr) {
      free(weight_r_ptr_);
      weight_r_ptr_ = nullptr;
    }
  }
  if (!is_vec_ || in_tensors_[3]->data_type() == kNumberTypeFloat32) {
    if (bias_ptr_ != nullptr) {
      free(bias_ptr_);
      bias_ptr_ = nullptr;
    }
  }
}

void GruFp16CPUKernel::FreeRunBuffer() {
  context_->allocator->Free(gate_buffer_);
  if (!is_vec_) {
    for (int i = 0; i < 2; i++) {
      context_->allocator->Free(matmul_buffer_[i]);
    }
  }
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
  gru_param_->hidden_size_ = w_shape.at(1) / 3;

  gru_param_->input_step_ = gru_param_->batch_ * gru_param_->input_size_;
  gru_param_->output_step_ = gru_param_->bidirectional_ ? 2 * gru_param_->batch_ * gru_param_->hidden_size_
                                                        : gru_param_->batch_ * gru_param_->hidden_size_;

  is_vec_ = gru_param_->batch_ == 1;
  gru_param_->row_align_ = is_vec_ ? gru_param_->batch_ : UP_ROUND(gru_param_->batch_, C16NUM);
  gru_param_->col_align_ = is_vec_ ? gru_param_->hidden_size_ : UP_ROUND(gru_param_->hidden_size_, C8NUM);
  return RET_OK;
}

int GruFp16CPUKernel::InitWeight(const lite::Tensor *tensor, float16_t *ptr, int deep) {
  auto weight_batch = gru_param_->bidirectional_ ? 6 : 3;
  if (tensor->data_type() == kNumberTypeFloat32) {
    auto weight_data = reinterpret_cast<float *>(tensor->data_c());
    is_vec_ ? Float32ToFloat16(weight_data, ptr, tensor->ElementsNum())
            : PackLstmWeightFp32ToFp16(ptr, weight_data, weight_batch, deep, gru_param_->hidden_size_,
                                       gru_param_->col_align_);
  } else if (tensor->data_type() == kNumberTypeFloat16) {
    auto weight_data = reinterpret_cast<float16_t *>(tensor->data_c());
    if (is_vec_) {
      ptr = weight_data;
    } else {
      PackLstmWeightFp16(ptr, weight_data, weight_batch, deep, gru_param_->hidden_size_, gru_param_->col_align_);
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::InitWeightBias() {
  auto weight_batch = gru_param_->bidirectional_ ? 6 : 3;
  // malloc and init input * weight right matrix buffer
  auto weight_g = in_tensors_.at(1);
  MS_ASSERT(weight_g != nullptr);
  if (!is_vec_ || weight_g->data_type() == kNumberTypeFloat32) {
    weight_g_ptr_ = reinterpret_cast<float16_t *>(
      malloc(weight_batch * gru_param_->col_align_ * gru_param_->input_size_ * sizeof(float16_t)));
    if (weight_g_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruFp16CPUKernel malloc weight_g_ptr_ error.";
      return RET_ERROR;
    }
  }
  auto ret = InitWeight(weight_g, weight_g_ptr_, gru_param_->input_size_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruFp16CPUKernel init weight_g failed.";
    return RET_ERROR;
  }

  // malloc and init state * weight right matrix buffer
  auto weight_r = in_tensors_.at(2);
  MS_ASSERT(weight_r != nullptr);
  if (!is_vec_ || weight_r->data_type() == kNumberTypeFloat32) {
    weight_r_ptr_ = reinterpret_cast<float16_t *>(
      malloc(weight_batch * gru_param_->col_align_ * gru_param_->hidden_size_ * sizeof(float16_t)));
    if (weight_r_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruFp16CPUKernel malloc weight_r_ptr_ error.";
      return RET_ERROR;
    }
  }
  ret = InitWeight(weight_r, weight_r_ptr_, gru_param_->hidden_size_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruFp16CPUKernel init weight_r failed.";
    return RET_ERROR;
  }

  int bias_batch = gru_param_->bidirectional_ ? 12 : 6;
  auto bias = in_tensors_.at(3);
  MS_ASSERT(bias != nullptr);
  if (!is_vec_ || bias->data_type() == kNumberTypeFloat32) {
    bias_ptr_ = reinterpret_cast<float16_t *>(malloc(bias_batch * gru_param_->col_align_ * sizeof(float16_t)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "GruFp16CPUKernel malloc bias_ptr_ error.";
      return RET_ERROR;
    }
    memset(bias_ptr_, 0, bias_batch * gru_param_->col_align_ * sizeof(float16_t));
  }
  if (bias->data_type() == kNumberTypeFloat32) {
    auto bias_data = reinterpret_cast<float *>(bias->data_c());
    for (int i = 0; i < bias_batch; i++) {
      auto src_batch = bias_data + i * gru_param_->hidden_size_;
      auto dst_batch = bias_ptr_ + i * gru_param_->col_align_;
      Float32ToFloat16(src_batch, dst_batch, gru_param_->hidden_size_);
    }
  } else if (bias->data_type() == kNumberTypeFloat16) {
    auto bias_data = reinterpret_cast<float16_t *>(bias->data_c());
    if (is_vec_) {
      bias_ptr_ = bias_data;
    } else {
      for (int i = 0; i < bias_batch; i++) {
        auto src_batch = bias_data + i * gru_param_->hidden_size_;
        auto dst_batch = bias_ptr_ + i * gru_param_->col_align_;
        memcpy(dst_batch, src_batch, gru_param_->hidden_size_ * sizeof(float16_t));
      }
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::Init() {
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
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruFp16CPUKernel InitWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::MallocRunBuffer() {
  if (!is_vec_) {
    matmul_buffer_[0] = reinterpret_cast<float16_t *>(
      context_->allocator->Malloc(3 * gru_param_->row_align_ * gru_param_->input_size_ * sizeof(float16_t)));
    if (matmul_buffer_[0] == nullptr) {
      MS_LOG(ERROR) << "GruFp16CPUKernel malloc input * weight left matirx error.";
      return RET_ERROR;
    }

    matmul_buffer_[1] = reinterpret_cast<float16_t *>(
      context_->allocator->Malloc(3 * gru_param_->row_align_ * gru_param_->hidden_size_ * sizeof(float16_t)));
    if (matmul_buffer_[1] == nullptr) {
      MS_LOG(ERROR) << "GruFp16CPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  gate_buffer_ = reinterpret_cast<float16_t *>(
    context_->allocator->Malloc(4 * gru_param_->batch_ * gru_param_->hidden_size_ * sizeof(float16_t)));
  if (gate_buffer_ == nullptr) {
    MS_LOG(ERROR) << "GruFp16CPUKernel malloc gate_buffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruFp16CPUKernel::Run() {
  auto input = in_tensors_.at(kInputIndex);
  MS_ASSERT(input != nullptr);
  auto hidden_state = in_tensors_.at(4);
  MS_ASSERT(hidden_state != nullptr);
  auto output = out_tensors_.at(0);
  MS_ASSERT(output != nullptr);
  auto input_ptr = reinterpret_cast<float16_t *>(input->data_c());
  MS_ASSERT(input_ptr);
  auto output_ptr = reinterpret_cast<float16_t *>(output->data_c());
  MS_ASSERT(output_ptr);
  auto output_hidden_state = out_tensors_[1];
  memcpy(output_hidden_state->data_c(), hidden_state->data_c(), hidden_state->ElementsNum() * sizeof(float16_t));
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
    MS_LOG(ERROR) << "GruFp16CPUKernel MallocRunBuffer error.";
    return RET_ERROR;
  }
  MS_ASSERT(weight_g_ptr_ != nullptr);
  MS_ASSERT(weight_r_ptr_ != nullptr);
  MS_ASSERT(bias_ptr_ != nullptr);
  MS_ASSERT(gate_buffer_ != nullptr);
  GruFp16(output_ptr, input_ptr, weight_g_ptr_, weight_r_ptr_, bias_ptr_,
          reinterpret_cast<float16_t *>(output_hidden_state->data_c()), gate_buffer_, matmul_buffer_, check_seq_len,
          gru_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_GRU, LiteKernelCreator<GruFp16CPUKernel>)
}  // namespace mindspore::kernel
