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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/lstm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Lstm;

namespace mindspore::kernel {
void LstmFp16CPUKernel::FreeTmpBuffer() {
  if (!is_vec_ || in_tensors_[1]->data_type() == kNumberTypeFloat32) {
    if (weight_i_ptr_ != nullptr) {
      free(weight_i_ptr_);
      weight_i_ptr_ = nullptr;
    }
  }
  if (!is_vec_ || in_tensors_[2]->data_type() == kNumberTypeFloat32) {
    if (weight_h_ptr_ != nullptr) {
      free(weight_h_ptr_);
      weight_h_ptr_ = nullptr;
    }
  }
  if (!is_vec_ || in_tensors_[3]->data_type() == kNumberTypeFloat32) {
    if (bias_ptr_ != nullptr) {
      free(bias_ptr_);
      bias_ptr_ = nullptr;
    }
  }
}

void LstmFp16CPUKernel::FreeRunBuffer() {
  context_->allocator->Free(gate_buffer_);
  context_->allocator->Free(state_buffer_);
  if (!is_vec_) {
    for (int i = 0; i < 2; i++) {
      context_->allocator->Free(matmul_buffer_[i]);
    }
  }
}

int LstmFp16CPUKernel::InitParam() {
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

  lstm_param_->input_step_ = lstm_param_->batch_ * lstm_param_->input_size_;
  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? 2 * lstm_param_->batch_ * lstm_param_->hidden_size_
                                                          : lstm_param_->batch_ * lstm_param_->hidden_size_;

  is_vec_ = lstm_param_->batch_ == 1;
  lstm_param_->row_align_ = is_vec_ ? lstm_param_->batch_ : UP_ROUND(lstm_param_->batch_, C16NUM);
  lstm_param_->col_align_ = is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, C8NUM);
  return RET_OK;
}

int LstmFp16CPUKernel::InitWeight(const lite::Tensor *tensor, float16_t *ptr, int deep) {
  auto weight_batch = lstm_param_->bidirectional_ ? 8 : 4;
  if (tensor->data_type() == kNumberTypeFloat32) {
    auto weight_data = reinterpret_cast<float *>(tensor->data_c());
    is_vec_ ? Float32ToFloat16(weight_data, ptr, tensor->ElementsNum())
            : PackLstmWeightFp32ToFp16(ptr, weight_data, weight_batch, deep, lstm_param_->hidden_size_,
                                       lstm_param_->col_align_);
  } else if (tensor->data_type() == kNumberTypeFloat16) {
    auto weight_data = reinterpret_cast<float16_t *>(tensor->data_c());
    if (is_vec_) {
      ptr = weight_data;
    } else {
      PackLstmWeightFp16(ptr, weight_data, weight_batch, deep, lstm_param_->hidden_size_, lstm_param_->col_align_);
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type of weight tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16CPUKernel::InitWeightBias() {
  auto weight_batch = lstm_param_->bidirectional_ ? 8 : 4;
  // malloc and init input * weight right matrix buffer
  auto weight_i = in_tensors_.at(1);
  MS_ASSERT(weight_i != nullptr);
  if (!is_vec_ || weight_i->data_type() == kNumberTypeFloat32) {
    weight_i_ptr_ = reinterpret_cast<float16_t *>(
      malloc(weight_batch * lstm_param_->col_align_ * lstm_param_->input_size_ * sizeof(float16_t)));
    if (weight_i_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc weight_i_ptr_ error.";
      return RET_ERROR;
    }
  }
  auto ret = InitWeight(weight_i, weight_i_ptr_, lstm_param_->input_size_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel init weight_i failed.";
    return RET_ERROR;
  }

  // malloc and init state * weight right matrix buffer
  auto weight_h = in_tensors_.at(2);
  MS_ASSERT(weight_h != nullptr);
  if (!is_vec_ || weight_h->data_type() == kNumberTypeFloat32) {
    weight_h_ptr_ = reinterpret_cast<float16_t *>(
      malloc(weight_batch * lstm_param_->col_align_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
    if (weight_h_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc weight_h_ptr_ error.";
      return RET_ERROR;
    }
  }
  ret = InitWeight(weight_h, weight_h_ptr_, lstm_param_->hidden_size_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel init weight_h failed.";
    return RET_ERROR;
  }

  int bias_batch = lstm_param_->bidirectional_ ? 16 : 8;
  auto bias = in_tensors_.at(3);
  MS_ASSERT(bias != nullptr);
  if (!is_vec_ || bias->data_type() == kNumberTypeFloat32) {
    bias_ptr_ = reinterpret_cast<float16_t *>(malloc(bias_batch * lstm_param_->col_align_ * sizeof(float16_t)));
    if (bias_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc bias_ptr_ error.";
      return RET_ERROR;
    }
    memset(bias_ptr_, 0, bias_batch * lstm_param_->col_align_ * sizeof(float16_t));
  }
  if (bias->data_type() == kNumberTypeFloat32) {
    auto bias_data = reinterpret_cast<float *>(bias->data_c());
    for (int i = 0; i < bias_batch; i++) {
      auto src_batch = bias_data + i * lstm_param_->hidden_size_;
      auto dst_batch = bias_ptr_ + i * lstm_param_->col_align_;
      Float32ToFloat16(src_batch, dst_batch, lstm_param_->hidden_size_);
    }
  } else if (bias->data_type() == kNumberTypeFloat16) {
    auto bias_data = reinterpret_cast<float16_t *>(bias->data_c());
    if (is_vec_) {
      bias_ptr_ = bias_data;
    } else {
      for (int i = 0; i < bias_batch; i++) {
        auto src_batch = bias_data + i * lstm_param_->hidden_size_;
        auto dst_batch = bias_ptr_ + i * lstm_param_->col_align_;
        memcpy(dst_batch, src_batch, lstm_param_->hidden_size_ * sizeof(float16_t));
      }
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type of bias tensor for lstm.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16CPUKernel::Init() {
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
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Lstm fp16 InitWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmFp16CPUKernel::MallocRunBuffer() {
  if (!is_vec_) {
    matmul_buffer_[0] = reinterpret_cast<float16_t *>(
      context_->allocator->Malloc(4 * lstm_param_->row_align_ * lstm_param_->input_size_ * sizeof(float16_t)));
    if (matmul_buffer_[0] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc input * weight left matirx error.";
      return RET_ERROR;
    }

    matmul_buffer_[1] = reinterpret_cast<float16_t *>(
      context_->allocator->Malloc(4 * lstm_param_->row_align_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
    if (matmul_buffer_[1] == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state * weight left matirx error.";
      return RET_ERROR;
    }
  }

  gate_buffer_ = reinterpret_cast<float16_t *>(
    context_->allocator->Malloc(8 * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
  if (gate_buffer_ == nullptr) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel malloc gate_buffer error.";
    return RET_ERROR;
  }
  if (!(lstm_param_->smooth_ >= -FLT_EPSILON && lstm_param_->smooth_ <= FLT_EPSILON)) {
    int buffer_size = 2 * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t);
    state_buffer_ = reinterpret_cast<float16_t *>(context_->allocator->Malloc(buffer_size));
    if (state_buffer_ == nullptr) {
      MS_LOG(ERROR) << "LstmFp16CPUKernel malloc state_buffer error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LstmFp16CPUKernel::Run() {
  auto input = in_tensors_.at(kInputIndex);
  MS_ASSERT(input != nullptr);
  auto hidden_state = in_tensors_.at(4);
  MS_ASSERT(hidden_state != nullptr);
  auto cell_state = in_tensors_.at(5);
  MS_ASSERT(cell_state != nullptr);
  auto output = out_tensors_.at(0);
  MS_ASSERT(output != nullptr);

  auto input_ptr = reinterpret_cast<float16_t *>(input->data_c());
  MS_ASSERT(input_ptr);
  auto output_ptr = reinterpret_cast<float16_t *>(output->data_c());
  MS_ASSERT(output_ptr);
  auto output_hidden_state = out_tensors_[1];
  memcpy(output_hidden_state->data_c(), hidden_state->data_c(), hidden_state->ElementsNum() * sizeof(float16_t));
  auto output_cell_state = out_tensors_[2];
  memcpy(output_cell_state->data_c(), cell_state->data_c(), cell_state->ElementsNum() * sizeof(float16_t));

  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmFp16CPUKernel MallocRunBuffer error.";
    return RET_ERROR;
  }
  MS_ASSERT(weight_i_ptr_);
  MS_ASSERT(weight_h_ptr_);
  MS_ASSERT(bias_ptr_);
  MS_ASSERT(gate_buffer_);
  LstmFp16(output_ptr, input_ptr, weight_i_ptr_, weight_h_ptr_, bias_ptr_,
           reinterpret_cast<float16_t *>(output_hidden_state->data_c()),
           reinterpret_cast<float16_t *>(output_cell_state->data_c()), gate_buffer_, state_buffer_, matmul_buffer_,
           lstm_param_);
  FreeRunBuffer();
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Lstm, LiteKernelCreator<LstmFp16CPUKernel>)
}  // namespace mindspore::kernel
