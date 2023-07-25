/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/lstm_non_mindir_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

namespace mindspore::kernel {
namespace {
constexpr int kInputGateIndex = 0;
constexpr int kGateNum = 4;
constexpr int kWeightInputIndex = 1;
constexpr int kWeightHiddenindex = 2;
constexpr int kCombinedBiasIndex = 3;
}  // namespace

int LstmNonMindirFp32CPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  weight_i_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(
    weight_segment_num_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float)));
  MS_CHECK_TRUE_MSG(weight_i_ptr_ != nullptr, lite::RET_NULL_PTR,
                    "LstmNonMindirCPUKernel malloc weight_i_ptr_ failed.");
  running_buffer_.push_back(weight_i_ptr_);
  auto weight_i = in_tensors_.at(kWeightInputIndex);
  auto weight_i_data = reinterpret_cast<float *>(weight_i->data());
  CHECK_NULL_RETURN(weight_i_data);

  int stride = kGateNum * lstm_param_->input_size_ * lstm_param_->hidden_size_;
  PackLstmWeightWithStride(weight_i_ptr_, weight_i_data, weight_segment_num_, lstm_param_->input_size_,
                           lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_,
                           stride, nullptr);
  // input bias
  input_bias_ = reinterpret_cast<float *>(
    ms_context_->allocator->Malloc(weight_segment_num_ * lstm_param_->input_col_align_ * sizeof(float)));
  MS_CHECK_TRUE_MSG(input_bias_ != nullptr, lite::RET_NULL_PTR, "LstmNonMindirCPUKernel malloc input_bias_ failed.");
  memset(input_bias_, 0, weight_segment_num_ * lstm_param_->input_col_align_ * sizeof(float));
  running_buffer_.push_back(input_bias_);
  auto bias_data = reinterpret_cast<float *>(in_tensors_.at(kCombinedBiasIndex)->data());
  CHECK_NULL_RETURN(bias_data);
  PackLstmBias(input_bias_, bias_data, weight_segment_num_, lstm_param_->hidden_size_, lstm_param_->input_col_align_,
               lstm_param_->bidirectional_, nullptr);
  return RET_OK;
}

int LstmNonMindirFp32CPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_h = in_tensors_.at(kWeightHiddenindex);
  auto weight_h_data = reinterpret_cast<float *>(weight_h->data());
  CHECK_NULL_RETURN(weight_h_data);

  int stride = kGateNum * lstm_param_->hidden_size_ * lstm_param_->output_size_;
  auto weight_pack_size =
    weight_segment_num_ * lstm_param_->state_col_align_ * lstm_param_->output_size_ * sizeof(float);
  if (lstm_param_->batch_ != 1) {
    weight_h_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(weight_pack_size));
    MS_CHECK_TRUE_MSG(weight_h_ptr_ != nullptr, lite::RET_NULL_PTR,
                      "LstmNonMindirCPUKernel malloc weight_h_ptr_ failed.");
    running_buffer_.push_back(weight_h_ptr_);
    PackLstmWeightWithStride(weight_h_ptr_, weight_h_data, weight_segment_num_, lstm_param_->output_size_,
                             lstm_param_->hidden_size_, lstm_param_->state_col_align_, lstm_param_->bidirectional_,
                             stride, nullptr);
  } else {
#ifdef ENABLE_AVX
    weight_h_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(weight_pack_size));
    MS_CHECK_TRUE_MSG(weight_h_ptr_ != nullptr, lite::RET_NULL_PTR,
                      "LstmNonMindirCPUKernel malloc weight_h_ptr_ failed.");
    running_buffer_.push_back(weight_h_ptr_);
    for (int i = 0; i < weight_segment_num_; i++) {
      const float *src_batch = weight_h_data + i * lstm_param_->hidden_size_ * lstm_param_->output_size_;
      float *dst_batch = weight_h_ptr_ + i * lstm_param_->state_col_align_ * lstm_param_->output_size_;
      RowMajor2Col32Major(src_batch, dst_batch, lstm_param_->hidden_size_, lstm_param_->output_size_);
    }
#else
    weight_h_ptr_ = weight_h_data;
#endif
  }

  // state bias
  auto bias_pack_size = weight_segment_num_ * lstm_param_->state_col_align_ * sizeof(float);
  state_bias_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(bias_pack_size));
  MS_CHECK_TRUE_MSG(state_bias_ != nullptr, lite::RET_NULL_PTR, "LstmNonMindirCPUKernel malloc state_bias_ failed.");
  memset(state_bias_, 0, bias_pack_size);
  running_buffer_.push_back(state_bias_);
  // if ONNX, secend bias is also present order IOFG
  auto bias_data = reinterpret_cast<float *>(in_tensors_.at(kCombinedBiasIndex)->data());
  CHECK_NULL_RETURN(bias_data);
  auto *state_bias = bias_data + kGateNum * lstm_param_->hidden_size_;
  PackLstmBias(state_bias_, state_bias, weight_segment_num_, lstm_param_->hidden_size_, lstm_param_->state_col_align_,
               lstm_param_->bidirectional_, nullptr);
  return RET_OK;
}

int LstmNonMindirFp32CPUKernel::InitProjectWeight() {
  if (in_tensors_.size() < C7NUM) {
    return RET_OK;
  }
  auto weight_pro = in_tensors_.at(SEVENTH_INPUT);
  auto shape = weight_pro->shape();
  MS_CHECK_TRUE_MSG(shape.size() == C3NUM, lite::RET_ERROR, "Project-weight's shape must be 3D.");
  auto weight_pro_data = reinterpret_cast<float *>(weight_pro->data());
  CHECK_NULL_RETURN(weight_pro_data);
  int batch = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  if (shape[0] != batch) {
    MS_LOG(ERROR) << "Project-weight's shape[0] must be 1(bidirectional=false) or 2(bidirectional=true).";
    return lite::RET_ERROR;
  }
  int col_align = UP_ROUND(lstm_param_->output_size_, col_tile_);
  auto pack_size = batch * lstm_param_->hidden_size_ * col_align * sizeof(float);
  if (lstm_param_->batch_ != 1) {
    weight_project_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_size));
    MS_CHECK_TRUE_MSG(weight_project_ptr_ != nullptr, lite::RET_NULL_PTR,
                      "LstmNonMindirCPUKernel malloc weight_project_ptr_ failed.");
    running_buffer_.push_back(weight_project_ptr_);
    PackLstmWeightWithStride(weight_project_ptr_, weight_pro_data, batch, lstm_param_->hidden_size_,
                             lstm_param_->output_size_, col_align, lstm_param_->bidirectional_,
                             lstm_param_->hidden_size_ * lstm_param_->output_size_, nullptr);
  } else {
#ifdef ENABLE_AVX
    weight_project_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_size));
    MS_CHECK_TRUE_MSG(weight_project_ptr_ != nullptr, lite::RET_NULL_PTR,
                      "LstmNonMindirCPUKernel malloc weight_project_ptr_ failed.");
    running_buffer_.push_back(weight_project_ptr_);
    for (int i = 0; i < batch; ++i) {
      const float *src_batch = weight_pro_data + i * lstm_param_->hidden_size_ * lstm_param_->output_size_;
      float *dst_batch = weight_project_ptr_ + i * lstm_param_->hidden_size_ * col_align;
      RowMajor2Col32Major(src_batch, dst_batch, lstm_param_->output_size_, lstm_param_->hidden_size_);
    }
#else
    weight_project_ptr_ = weight_pro_data;
#endif
  }
  return RET_OK;
}

void LstmNonMindirFp32CPUKernel::LstmUnidirectional(float *output, const float *weight_h, const float *state_bias,
                                                    float *hidden_state, float *cell_state, const float *weight_project,
                                                    float *intermediate_states, float **buffer, bool is_backward) {
  float *gate = buffer[kInputGateIndex];
  float *input_gate = gate;
  float *forget_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * C2NUM;
  float *cell_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * C3NUM;
  float *output_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_;
  for (int t = 0; t < lstm_param_->seq_len_; t++) {
    int real_t = is_backward ? lstm_param_->seq_len_ - t - C1NUM : t;
    float *input_gate_t = input_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *forget_gate_t = forget_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *cell_gate_t = cell_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *output_gate_t = output_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    // Sequence, DirMul, Batch, Hidden
    float *output_ptr = output + real_t * lstm_param_->output_step_;
    LstmStepUnit(output_ptr, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias,
                 weight_project, hidden_state, cell_state, buffer, lstm_param_);
  }
}
}  // namespace mindspore::kernel
