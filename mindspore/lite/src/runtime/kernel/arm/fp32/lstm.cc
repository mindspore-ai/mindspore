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

#include "src/runtime/kernel/arm/fp32/lstm.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Lstm;

namespace mindspore::kernel {
int LstmCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  lstm_parm_->seq_len_ = in_shape[0];
  lstm_parm_->batch_ = in_shape[1];
  lstm_parm_->input_size_ = in_shape[2];

  auto weight_i = in_tensors_[1];
  MS_ASSERT(weight_i != nullptr);
  std::vector<int> w_shape = weight_i->shape();
  lstm_parm_->hidden_size_ = w_shape[1] / 4;

  lstm_parm_->input_step_ = lstm_parm_->batch_ * lstm_parm_->input_size_;
  lstm_parm_->output_step_ = lstm_parm_->bidirectional_ ? 2 * lstm_parm_->batch_ * lstm_parm_->hidden_size_
                                                        : lstm_parm_->batch_ * lstm_parm_->hidden_size_;
  return RET_OK;
}

int LstmCPUKernel::InitBuffer() {
  gate_buffer_ = reinterpret_cast<float *>(malloc(4 * lstm_parm_->batch_ * lstm_parm_->hidden_size_ * sizeof(float)));
  if (gate_buffer_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc gate_buffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmCPUKernel::InitWeightBias() {
  // copy weight_i and weight_h
  auto weight_i = in_tensors_.at(1);
  MS_ASSERT(weight_i != nullptr);
  weight_i_ptr_ = reinterpret_cast<float *>(malloc(weight_i->ElementsNum() * sizeof(float)));
  if (weight_i_ptr_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc weight_i_ptr_ error.";
    return RET_ERROR;
  }
  memcpy(weight_i_ptr_, weight_i->Data(), weight_i->ElementsNum() * sizeof(float));

  auto weight_h = in_tensors_.at(2);
  MS_ASSERT(weight_h != nullptr);
  weight_h_ptr_ = reinterpret_cast<float *>(malloc(weight_h->ElementsNum() * sizeof(float)));
  if (weight_h_ptr_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc weight_h_ error.";
    return RET_ERROR;
  }
  memcpy(weight_h_ptr_, weight_h->Data(), weight_h->ElementsNum() * sizeof(float));

  // init bias
  int bias_num = lstm_parm_->bidirectional_ ? 2 * 4 * lstm_parm_->hidden_size_ : 4 * lstm_parm_->hidden_size_;
  bias_ptr_ = reinterpret_cast<float *>(malloc(bias_num * sizeof(float)));
  if (bias_ptr_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc bias_ptr_ error.";
    return RET_ERROR;
  }

  auto bias_data = reinterpret_cast<float *>(in_tensors_.at(3)->Data());
  const int state_bias_offset = 4 * lstm_parm_->hidden_size_;
  for (int i = 0; i < state_bias_offset; i++) {
    bias_ptr_[i] = bias_data[i] + bias_data[i + state_bias_offset];
  }
  if (lstm_parm_->bidirectional_) {
    bias_data += 4 * lstm_parm_->hidden_size_ * 2;
    auto backward_bias = bias_ptr_ + 4 * lstm_parm_->hidden_size_;
    for (int i = 0; i < state_bias_offset; i++) {
      backward_bias[i] = bias_data[i] + bias_data[i + state_bias_offset];
    }
  }
  return RET_OK;
}

int LstmCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitParam error.";
    return RET_ERROR;
  }

  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitWeightBias error.";
    return RET_ERROR;
  }

  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitBuffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmCPUKernel::ReSize() {
  free(gate_buffer_);

  auto ret = InitParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitParam error.";
    return RET_ERROR;
  }

  ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitBuffer error.";
    return RET_ERROR;
  }
  return RET_OK;
}

int LstmCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input = in_tensors_.at(kInputIndex);
  MS_ASSERT(input != nullptr);
  auto hidden_state = in_tensors_.at(4);
  MS_ASSERT(hidden_state != nullptr);
  auto cell_state = in_tensors_.at(5);
  MS_ASSERT(cell_state != nullptr);
  auto output = out_tensors_.at(0);
  MS_ASSERT(output != nullptr);

  auto input_ptr = reinterpret_cast<float *>(input->Data());
  auto output_ptr = reinterpret_cast<float *>(output->Data());

  auto output_hidden_state = out_tensors_[1];
  memcpy(output_hidden_state->Data(), hidden_state->Data(), hidden_state->ElementsNum() * sizeof(float));
  auto output_cell_state = out_tensors_[2];
  memcpy(output_cell_state->Data(), cell_state->Data(), cell_state->ElementsNum() * sizeof(float));

  Lstm(output_ptr, input_ptr, weight_i_ptr_, weight_h_ptr_, bias_ptr_,
       reinterpret_cast<float *>(output_hidden_state->Data()), reinterpret_cast<float *>(output_cell_state->Data()),
       gate_buffer_, lstm_parm_);
  return RET_OK;
}

kernel::LiteKernel *CpuLstmKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                         const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *opParameter,
                                         const lite::Context *ctx, const kernel::KernelKey &desc,
                                         const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Lstm);

  auto *kernel = new (std::nothrow) LstmCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Lstm, CpuLstmKernelCreator)
}  // namespace mindspore::kernel
