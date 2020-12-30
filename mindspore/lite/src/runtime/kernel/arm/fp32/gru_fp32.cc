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

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gru;

namespace mindspore::kernel {
void GruCPUKernel::FreeTmpBuffer() {
  if (gate_buffer_ != nullptr) {
    free(gate_buffer_);
    gate_buffer_ = nullptr;
  }
  if (bias_ptr_ != nullptr) {
    free(bias_ptr_);
    bias_ptr_ = nullptr;
  }
  weight_g_ptr_ = nullptr;
  weight_r_ptr_ = nullptr;
}

int GruCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  gru_parm_->seq_len_ = in_shape.at(0);
  gru_parm_->batch_ = in_shape.at(1);
  gru_parm_->input_size_ = in_shape.at(2);

  auto weight_g = in_tensors_.at(1);
  MS_ASSERT(weight_g != nullptr);
  std::vector<int> w_shape = weight_g->shape();
  gru_parm_->hidden_size_ = w_shape.at(1) / 3;

  gru_parm_->input_step_ = gru_parm_->batch_ * gru_parm_->input_size_;
  gru_parm_->output_step_ = gru_parm_->bidirectional_ ? 2 * gru_parm_->batch_ * gru_parm_->hidden_size_
                                                      : gru_parm_->batch_ * gru_parm_->hidden_size_;
  return RET_OK;
}

int GruCPUKernel::InitBuffer() {
  gate_buffer_ = reinterpret_cast<float *>(malloc(3 * gru_parm_->batch_ * gru_parm_->hidden_size_ * sizeof(float)));
  if (gate_buffer_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc gate_buffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GruCPUKernel::InitWeightBias() {
  auto weight_gate = in_tensors_.at(1);
  MS_ASSERT(weight_gate != nullptr);
  weight_g_ptr_ = reinterpret_cast<float *>(weight_gate->data_c());

  auto weight_recu = in_tensors_.at(2);
  MS_ASSERT(weight_recu != nullptr);
  weight_r_ptr_ = reinterpret_cast<float *>(weight_recu->data_c());

  int bias_num = gru_parm_->bidirectional_ ? 2 * 3 * gru_parm_->hidden_size_ : 3 * gru_parm_->hidden_size_;
  bias_ptr_ = reinterpret_cast<float *>(malloc(bias_num * sizeof(float)));
  if (bias_ptr_ == nullptr) {
    MS_LOG(ERROR) << "GruCPUKernel malloc bias_ptr_ error.";
    return RET_ERROR;
  }

  auto bias_data = reinterpret_cast<float *>(in_tensors_.at(3)->data_c());
  const int state_bias_offset = 3 * gru_parm_->hidden_size_;
  for (int i = 0; i < state_bias_offset; i++) {
    bias_ptr_[i] = bias_data[i] + bias_data[i + state_bias_offset];
  }
  if (gru_parm_->bidirectional_) {
    bias_data += 3 * gru_parm_->hidden_size_ * 2;
    auto backward_bias = bias_ptr_ + 3 * gru_parm_->hidden_size_;
    for (int i = 0; i < state_bias_offset; i++) {
      backward_bias[i] = bias_data[i] + bias_data[i + state_bias_offset];
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
  FreeTmpBuffer();
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel InitParam error.";
    return RET_ERROR;
  }

  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel InitWeightBias error.";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GruCPUKernel InitBuffer error.";
    FreeTmpBuffer();
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
  int check_seq_len = gru_parm_->seq_len_;
  if (in_tensors_.size() == 6) {
    auto seq_len = reinterpret_cast<int *>(in_tensors_.at(5)->data_c());
    if (!std::equal(seq_len + 1, seq_len + gru_parm_->batch_, seq_len)) {
      MS_LOG(ERROR) << "different batch seq_len is currently not supported";
      return RET_ERROR;
    }
    check_seq_len = MSMIN(check_seq_len, MSMAX(0, seq_len[0]));
  }

  MS_ASSERT(weight_g_ptr_ != nullptr);
  MS_ASSERT(weight_r_ptr_ != nullptr);
  MS_ASSERT(bias_ptr_ != nullptr);
  MS_ASSERT(gate_buffer_ != nullptr);
  Gru(output_ptr, input_ptr, weight_g_ptr_, weight_r_ptr_, bias_ptr_,
      reinterpret_cast<float *>(output_hidden_state->data_c()), gate_buffer_, check_seq_len, gru_parm_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Gru, LiteKernelCreator<GruCPUKernel>)
}  // namespace mindspore::kernel
