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

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Lstm;

namespace mindspore::kernel {
void LstmFp16CPUKernel::FreeTmpBuffer() {
  if (gate_buffer_ != nullptr) {
    free(gate_buffer_);
    gate_buffer_ = nullptr;
  }
  if (state_buffer_ != nullptr) {
    free(state_buffer_);
    state_buffer_ = nullptr;
  }
  if (weight_i_ptr_ != nullptr) {
    free(weight_i_ptr_);
    weight_i_ptr_ = nullptr;
  }
  if (weight_h_ptr_ != nullptr) {
    free(weight_h_ptr_);
    weight_h_ptr_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
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
  return RET_OK;
}

int LstmFp16CPUKernel::InitBuffer() {
  gate_buffer_ =
    reinterpret_cast<float16_t *>(malloc(4 * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t)));
  if (gate_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Lstm fp16 malloc gate_buffer error.";
    return RET_ERROR;
  }
  if (!(lstm_param_->smooth_ >= -FLT_EPSILON && lstm_param_->smooth_ <= FLT_EPSILON)) {
    int buffer_size = 2 * lstm_param_->batch_ * lstm_param_->hidden_size_ * sizeof(float16_t);
    state_buffer_ = reinterpret_cast<float16_t *>(malloc(buffer_size));
    if (state_buffer_ == nullptr) {
      MS_LOG(ERROR) << "Lstm fp16 malloc state_buffer error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LstmFp16CPUKernel::InitWeightBias() {
  // copy weight_i and weight_h
  auto weight_i = in_tensors_.at(1);
  MS_ASSERT(weight_i != nullptr);
  weight_i_ptr_ = reinterpret_cast<float16_t *>(malloc(weight_i->ElementsNum() * sizeof(float16_t)));
  if (weight_i_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Lstm fp16 malloc weight_i_ptr_ error.";
    return RET_ERROR;
  }
  auto weight_i_data = reinterpret_cast<float *>(weight_i->data_c());
  for (size_t i = 0; i < weight_i->ElementsNum(); i++) {
    weight_i_ptr_[i] = (float16_t)weight_i_data[i];
  }

  auto weight_h = in_tensors_.at(2);
  MS_ASSERT(weight_h != nullptr);
  weight_h_ptr_ = reinterpret_cast<float16_t *>(malloc(weight_h->ElementsNum() * sizeof(float16_t)));
  if (weight_h_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Lstm fp16 malloc weight_h_ error.";
    return RET_ERROR;
  }
  auto weight_h_data = reinterpret_cast<float *>(weight_h->data_c());
  for (size_t i = 0; i < weight_h->ElementsNum(); i++) {
    weight_h_ptr_[i] = (float16_t)weight_h_data[i];
  }

  std::vector<int> w_shape = weight_i->shape();
  auto hidden_size = w_shape.at(1) / 4;
  // init bias
  int bias_num = lstm_param_->bidirectional_ ? 2 * 4 * hidden_size : 4 * hidden_size;
  bias_ptr_ = reinterpret_cast<float16_t *>(malloc(bias_num * sizeof(float16_t)));
  if (bias_ptr_ == nullptr) {
    MS_LOG(ERROR) << "Lstm fp16 malloc bias_ptr_ error.";
    return RET_ERROR;
  }

  auto bias_data = reinterpret_cast<float *>(in_tensors_.at(3)->data_c());
  const int state_bias_offset = 4 * hidden_size;
  for (int i = 0; i < state_bias_offset; i++) {
    bias_ptr_[i] = (float16_t)(bias_data[i] + bias_data[i + state_bias_offset]);
  }
  if (lstm_param_->bidirectional_) {
    bias_data += 4 * hidden_size * 2;
    auto backward_bias = bias_ptr_ + 4 * hidden_size;
    for (int i = 0; i < state_bias_offset; i++) {
      backward_bias[i] = (float16_t)(bias_data[i] + bias_data[i + state_bias_offset]);
    }
  }
  return RET_OK;
}

int LstmFp16CPUKernel::Init() {
  FreeTmpBuffer();
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Lstm fp16 InitWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }

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

  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Lstm fp16 InitBuffer error.";
    FreeTmpBuffer();
    return RET_ERROR;
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

  MS_ASSERT(weight_h_ptr_);
  MS_ASSERT(weight_i_ptr_);
  MS_ASSERT(bias_ptr_);
  MS_ASSERT(gate_buffer_);
  LstmFp16(output_ptr, input_ptr, weight_i_ptr_, weight_h_ptr_, bias_ptr_,
           reinterpret_cast<float16_t *>(output_hidden_state->data_c()),
           reinterpret_cast<float16_t *>(output_cell_state->data_c()), gate_buffer_, state_buffer_, lstm_param_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Lstm, LiteKernelCreator<LstmFp16CPUKernel>)
}  // namespace mindspore::kernel
