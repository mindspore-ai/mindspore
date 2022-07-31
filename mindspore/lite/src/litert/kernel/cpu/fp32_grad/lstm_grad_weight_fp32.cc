/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32_grad/lstm_grad_weight_fp32.h"
#include <string>
#include <memory>
#include "utils/ms_utils.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/lstm_fp32.h"

namespace mindspore {
namespace kernel {
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LSTMGradWeight;

int LSTMGradWeightCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_5D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LSTMGradWeightCPUKernel::ReSize() { return InitParam(); }

int LSTMGradWeightCPUKernel::Run() {
  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTMGradWeightCPUKernel MallocRunBuffer error.";
    FreeRunBuffer();
    return RET_ERROR;
  }

  auto seq_stride = lstm_param_->seq_len_ * lstm_param_->output_step_;
  auto input_tensor = in_tensors_.at(input_index);
  MS_ASSERT(input_tensor != nullptr);
  auto hidden_input_tensor = in_tensors_.at(hidden_input_index);
  MS_ASSERT(hidden_input_tensor != nullptr);
  auto intermediate_tensor = in_tensors_.at(intermediate_data_index);
  MS_ASSERT(intermediate_tensor != nullptr);

  // Get output tensors
  auto dW_tensor = out_tensors_.at(dW_out_index);
  MS_ASSERT(dW_tensor != nullptr);

  // Get Tensors Data
  int time_stamp_len = lstm_param_->batch_ * lstm_param_->hidden_size_;
  input_ = reinterpret_cast<float *>(input_tensor->data());
  memset(dW_tmp_, 0, dW_tensor->Size());

  int w_size = lstm_param_->hidden_size_ * lstm_param_->input_size_;
  int h_size = lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  int b_size = lstm_param_->hidden_size_;

  if (lstm_param_->bidirectional_) {
    // Adjust pointer to backward cell
    hidden_input_data_ = reinterpret_cast<float *>(hidden_input_tensor->data()) + time_stamp_len;
    intermediate_data_ = reinterpret_cast<float *>(intermediate_tensor->data());
    dA_ = intermediate_data_ + seq_stride * 1 + lstm_param_->seq_len_ * num_of_gates * time_stamp_len;
    intermediate_data_ += time_stamp_len;

    int w_offset = num_of_gates * (w_size + h_size);
    int v_offset = weight_batch_ * w_size + num_of_gates * h_size;
    int b_offset = weight_batch_ * (w_size + h_size) + num_of_gates * b_size;
    float *dw = dW_tmp_ + w_offset;
    float *dv = dW_tmp_ + v_offset;
    float *db = nullptr;
    if (lstm_param_->has_bias_) {
      db = dW_tmp_ + b_offset;
    }
    LstmBackpropUnidirectional(true, dw, dv, db);
  }
  // adjust to forward cell
  hidden_input_data_ = reinterpret_cast<float *>(hidden_input_tensor->data());
  intermediate_data_ = reinterpret_cast<float *>(intermediate_tensor->data());
  dA_ = intermediate_data_ + seq_stride * 1;
  int w_offset = 0;
  int v_offset = num_of_gates * w_size;
  int b_offset = weight_batch_ * (w_size + h_size);
  float *dw = dW_tmp_ + w_offset;
  float *dv = dW_tmp_ + v_offset;
  float *db = nullptr;
  if (lstm_param_->has_bias_) {
    db = dW_tmp_ + b_offset;
  }
  LstmBackpropUnidirectional(false, dw, dv, db);

  // setup output tensors
  dW_ = reinterpret_cast<float *>(dW_tensor->data());
  ReorderLstmWeightGrad(dW_, dW_tmp_, lstm_param_);
  FreeRunBuffer();
  return RET_OK;
}

int LSTMGradWeightCPUKernel::LstmBackpropUnidirectional(bool is_backward, float *dw, float *dv, float *db) {
  float *hidden_state = intermediate_data_;
  int state_len = lstm_param_->batch_ * lstm_param_->hidden_size_;
  int prev_time_stamp_offset = (is_backward) ? 1 : -1;
  int first_time_stamp = (is_backward) ? lstm_param_->seq_len_ - 1 : 0;

  for (int t = lstm_param_->seq_len_ - 1; t >= 0; t--) {
    int real_t = is_backward ? lstm_param_->seq_len_ - t - 1 : t;
    float *prev_hidden_state = (real_t == first_time_stamp)
                                 ? hidden_input_data_
                                 : hidden_state + (real_t + prev_time_stamp_offset) * lstm_param_->output_step_;
    float *curr_input = input_ + real_t * lstm_param_->batch_ * lstm_param_->input_size_;
    float *curr_da = dA_ + real_t * num_of_gates * state_len;
    LstmGradDoWeightStep(curr_input, prev_hidden_state, curr_da, dw, dv, db, workspace_, lstm_param_);
  }
  return RET_OK;
}

void LSTMGradWeightCPUKernel::ReorderLstmWeightGrad(float *dst, float *src, LstmGradParameter *param) {
  int uni_batch = param->bidirectional_ ? weight_batch_ / C2NUM : weight_batch_;
  // 4xWixWh,4xWirxWhr,4xBiBh,4xBirBhr
  ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->input_size_, getLstmOrderIOFG());
  src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->hidden_size_, getLstmOrderIOFG());
  src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  if (param->bidirectional_) {
    ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->input_size_, getLstmOrderIOFG());
    src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
    dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->input_size_;
    ReorderLstmWeights(dst, src, uni_batch, lstm_param_->hidden_size_, lstm_param_->hidden_size_, getLstmOrderIOFG());
    src += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
    dst += uni_batch * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  }
  if (param->has_bias_) {
    ReorderLstmWeights(dst, src, uni_batch, 1, lstm_param_->hidden_size_, getLstmOrderIOFG());
    dst += uni_batch * lstm_param_->hidden_size_;
    ReorderLstmWeights(dst, src, uni_batch, 1, lstm_param_->hidden_size_, getLstmOrderIOFG());
    dst += uni_batch * lstm_param_->hidden_size_;
    src += uni_batch * lstm_param_->hidden_size_;
    if (param->bidirectional_) {
      ReorderLstmWeights(dst, src, uni_batch, 1, lstm_param_->hidden_size_, getLstmOrderIOFG());
      dst += uni_batch * lstm_param_->hidden_size_;
      ReorderLstmWeights(dst, src, uni_batch, 1, lstm_param_->hidden_size_, getLstmOrderIOFG());
      dst += uni_batch * lstm_param_->hidden_size_;
      src += uni_batch * lstm_param_->hidden_size_;
    }
  }
}

int LSTMGradWeightCPUKernel::DoGrad(int thread_id) { return RET_OK; }

int LSTMGradWeightCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  lstm_param_->seq_len_ = in_shape.at(FIRST_INPUT);
  lstm_param_->batch_ = in_shape.at(SECOND_INPUT);
  lstm_param_->input_size_ = in_shape.at(THIRD_INPUT);

  int dir_multiplier = lstm_param_->bidirectional_ ? 2 : 1;
  lstm_param_->output_step_ = dir_multiplier * lstm_param_->batch_ * lstm_param_->hidden_size_;
  weight_batch_ = dir_multiplier * num_of_gates;
  state_is_vec_ = lstm_param_->batch_ == 1;

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
  input_size_align_ = UP_ROUND(lstm_param_->input_size_, row_tile_);
  input_thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(lstm_param_->input_col_align_, col_tile_));
  input_thread_stride_ = UP_DIV(UP_DIV(lstm_param_->input_col_align_, col_tile_), input_thread_count_);

  state_row_tile_ = row_tile_;
  state_col_tile_ = col_tile_;

  lstm_param_->state_row_align_ = state_is_vec_ ? 1 : UP_ROUND(lstm_param_->batch_, state_row_tile_);
  lstm_param_->state_col_align_ =
    state_is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, state_col_tile_);

  return RET_OK;
}

int LSTMGradWeightCPUKernel::MallocRunBuffer() {
  int workspace_size = GetRunWorkspaceSize(lstm_param_);
  if (workspace_size == 0) {
    MS_LOG(ERROR) << "LstmGradWeightCPUKernel malloc run workspace 0 error.";
    return RET_ERROR;
  }

  workspace_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(workspace_size * sizeof(float)));
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "LstmGradWeightCPUKernel malloc run workspace error.";
    return RET_ERROR;
  }
  auto dW_tensor = out_tensors_.at(dW_out_index);
  MS_ASSERT(dW_tensor != nullptr);
  auto dW_size = dW_tensor->Size();
  if (dW_size == 0) {
    MS_LOG(ERROR) << "LstmGradWeightCPUKernel malloc run dW_tmp size error.";
    return RET_ERROR;
  }
  dW_tmp_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(dW_size));
  if (dW_tmp_ == nullptr) {
    MS_LOG(ERROR) << "LstmGradWeightCPUKernel malloc run dW_tmp alloc error.";
    return RET_ERROR;
  }
  return RET_OK;
}

void LSTMGradWeightCPUKernel::FreeRunBuffer() {
  if (workspace_ != nullptr) {
    ms_context_->allocator->Free(workspace_);
    workspace_ = nullptr;
  }
  if (dW_tmp_ != nullptr) {
    ms_context_->allocator->Free(dW_tmp_);
    dW_tmp_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LSTMGradWeight, LiteKernelCreator<LSTMGradWeightCPUKernel>)
}  // namespace kernel
}  // namespace mindspore
