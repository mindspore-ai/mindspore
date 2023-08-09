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

#include "src/litert/kernel/cpu/fp32/lstm_mindir_fp32.h"
#include "nnacl/fp32/pack_fp32.h"

namespace mindspore::kernel {
namespace {
constexpr int kInputGateIndex = 0;
constexpr int kTempHiddenOutputIndex = 8;
constexpr int kGateNum = 4;
constexpr int kWeightsIndex = 3;
const int kWeightsOrderMap[8] = {0, 2, 3, 1, 4, 6, 7, 5};  // IFGO order to IOFG order
}  // namespace

int LstmMindirFp32CPUKernel::ReSize() {
  auto ret = LstmFp32BaseCPUKernel::ReSize();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "LstmMindirFp32CPUKernel resize failed.";
    return ret;
  }
  // determine FB origin
  gpu_orig_state_ = false;
  auto weight_t = in_tensors_.at(kWeightsIndex);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(lstm_param_->hidden_size_, lstm_param_->input_size_, lite::RET_ERROR);
  int hi_unit_size = lstm_param_->hidden_size_ * lstm_param_->input_size_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_segment_num_, hi_unit_size, lite::RET_ERROR);
  int hi_whole_size = weight_segment_num_ * hi_unit_size;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(lstm_param_->hidden_size_, lstm_param_->output_size_, lite::RET_ERROR);
  int hh_unit_size = lstm_param_->hidden_size_ * lstm_param_->output_size_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_segment_num_, hh_unit_size, lite::RET_ERROR);
  int hh_whole_size = weight_segment_num_ * hh_unit_size;
  int scale = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(lstm_param_->hidden_size_, lstm_param_->project_size_, lite::RET_ERROR);
  int hp_unit_size = lstm_param_->hidden_size_ * lstm_param_->project_size_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(scale, hp_unit_size, lite::RET_ERROR);
  int hp_whole_size = scale * hp_unit_size;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_segment_num_ * C2NUM, lstm_param_->hidden_size_, lite::RET_ERROR);
  int bias_whole_size = weight_segment_num_ * C2NUM * lstm_param_->hidden_size_;
  auto whole_size = weight_t->ElementsNum();
  bool has_bias = (hi_whole_size + hh_whole_size + hp_whole_size < whole_size) ? true : false;
  // if bias exist we can determine the gpu_orig_state_
  if (has_bias) {
    gpu_orig_state_ = (hi_whole_size + hh_whole_size + hp_whole_size + bias_whole_size == whole_size) ? true : false;
  } else {
    bias_whole_size = 0;
  }
  if (gpu_orig_state_) {
    return lite::RET_OK;
  }
  bias_whole_size /= C2NUM;
  if (hi_whole_size + hh_whole_size + hp_whole_size + bias_whole_size != whole_size) {
    MS_LOG(ERROR) << "LstmMindir is invalid when original model exports from CPU.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  return lite::RET_OK;
}

int LstmMindirFp32CPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  weight_i_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(
    weight_segment_num_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float)));
  MS_CHECK_TRUE_MSG(weight_i_ptr_ != nullptr, lite::RET_NULL_PTR, "LstmMindirCPUKernel malloc weight_i_ptr_ failed.");
  running_buffer_.push_back(weight_i_ptr_);
  auto weight_data = reinterpret_cast<float *>(in_tensors_.at(kWeightsIndex)->data());
  CHECK_NULL_RETURN(weight_data);

  int hi_unit_size = lstm_param_->input_size_ * lstm_param_->hidden_size_;
  int hh_unit_size = lstm_param_->hidden_size_ * lstm_param_->output_size_;
  int stride = (gpu_orig_state_) ? kGateNum * (hi_unit_size + hh_unit_size) : kGateNum * hi_unit_size;
  PackLstmWeightWithStride(weight_i_ptr_, weight_data, weight_segment_num_, lstm_param_->input_size_,
                           lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_,
                           stride, kWeightsOrderMap);
  // input bias
  auto bias_size = weight_segment_num_ * lstm_param_->input_col_align_ * sizeof(float);
  input_bias_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(bias_size));
  MS_CHECK_TRUE_MSG(input_bias_ != nullptr, lite::RET_NULL_PTR, "LstmMindirCPUKernel malloc input_bias_ failed.");
  memset(input_bias_, 0, bias_size);
  running_buffer_.push_back(input_bias_);
  if (!lstm_param_->has_bias_) {
    return RET_OK;
  }
  int scale = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  int offset = weight_segment_num_ * (hi_unit_size + hh_unit_size) +
               scale * lstm_param_->project_size_ * lstm_param_->hidden_size_;
  float *bias_data = weight_data + offset;
  int b_stride =
    (gpu_orig_state_) ? kGateNum * (scale * lstm_param_->hidden_size_) : kGateNum * (lstm_param_->hidden_size_);
  PackLstmBiasWithStride(input_bias_, bias_data, weight_segment_num_, lstm_param_->hidden_size_,
                         lstm_param_->input_col_align_, lstm_param_->bidirectional_, b_stride, kWeightsOrderMap);
  return RET_OK;
}

int LstmMindirFp32CPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  auto weight_data = (reinterpret_cast<float *>(in_tensors_.at(kWeightsIndex)->data()));
  CHECK_NULL_RETURN(weight_data);

  int hi_unit_size = lstm_param_->input_size_ * lstm_param_->hidden_size_;
  int hh_unit_size = lstm_param_->hidden_size_ * lstm_param_->output_size_;
  int stride = (gpu_orig_state_) ? kGateNum * (hi_unit_size + hh_unit_size) : kGateNum * hh_unit_size;

  auto weight_h_data = weight_data + (gpu_orig_state_ ? kGateNum * hi_unit_size : weight_segment_num_ * hi_unit_size);

  auto weight_unit_pack_size = sizeof(float) * lstm_param_->state_col_align_ * lstm_param_->output_size_;
  auto weight_pack_size = weight_segment_num_ * weight_unit_pack_size;
  weight_h_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(weight_pack_size));
  MS_CHECK_TRUE_MSG(weight_h_ptr_ != nullptr, lite::RET_NULL_PTR, "LstmMindirCPUKernel malloc weight_h_ptr_ failed.");
  running_buffer_.push_back(weight_h_ptr_);
  if (lstm_param_->batch_ != 1) {
    PackLstmWeightWithStride(weight_h_ptr_, weight_h_data, weight_segment_num_, lstm_param_->output_size_,
                             lstm_param_->hidden_size_, lstm_param_->state_col_align_, lstm_param_->bidirectional_,
                             stride, kWeightsOrderMap);
  } else {
    for (int i = 0; i < weight_segment_num_; i++) {
      const float *src_batch = weight_h_data + i * lstm_param_->hidden_size_ * lstm_param_->output_size_;
      float *dst_batch =
        weight_h_ptr_ + kWeightsOrderMap[i] * lstm_param_->state_col_align_ * lstm_param_->output_size_;
#ifdef ENABLE_AVX
      RowMajor2Col32Major(src_batch, dst_batch, lstm_param_->hidden_size_, lstm_param_->output_size_);
#else
      (void)memcpy(dst_batch, src_batch, weight_unit_pack_size);
#endif
    }
  }

  // state bias
  auto bias_pack_size = weight_segment_num_ * lstm_param_->state_col_align_ * sizeof(float);
  state_bias_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(bias_pack_size));
  MS_CHECK_TRUE_MSG(state_bias_ != nullptr, lite::RET_NULL_PTR, "LstmMindirCPUKernel malloc state_bias_ failed.");
  memset(state_bias_, 0, bias_pack_size);
  running_buffer_.push_back(state_bias_);
  if (!lstm_param_->has_bias_ || !gpu_orig_state_) {
    return RET_OK;
  }

  int hi_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  int hh_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->output_size_;
  int proj_size =
    (lstm_param_->bidirectional_ ? C2NUM : C1NUM) * lstm_param_->project_size_ * lstm_param_->hidden_size_;
  // mindir from device "GPU", secend bias is also present order IFOG
  int bias_offset = hi_whole_size + hh_whole_size + proj_size + lstm_param_->hidden_size_ * kGateNum;
  float *state_bias = weight_data + bias_offset;
  int b_stride = kGateNum * lstm_param_->hidden_size_ * C2NUM;
  PackLstmBiasWithStride(state_bias_, state_bias, weight_segment_num_, lstm_param_->hidden_size_,
                         lstm_param_->state_col_align_, lstm_param_->bidirectional_, b_stride, kWeightsOrderMap);
  return RET_OK;
}

int LstmMindirFp32CPUKernel::InitProjectWeight() {
  if (lstm_param_->project_size_ == 0) {
    return RET_OK;
  }
  auto weight_data = (reinterpret_cast<float *>(in_tensors_.at(kWeightsIndex)->data()));
  CHECK_NULL_RETURN(weight_data);
  int hi_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  int hh_whole_size = weight_segment_num_ * lstm_param_->hidden_size_ * lstm_param_->output_size_;
  auto weight_proj_data = weight_data + hi_whole_size + hh_whole_size;
  int batch = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  auto pack_size = batch * lstm_param_->hidden_size_ * lstm_param_->proj_col_align_ * sizeof(float);
  if (lstm_param_->batch_ != 1) {
    weight_project_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_size));
    MS_CHECK_TRUE_MSG(weight_project_ptr_ != nullptr, lite::RET_NULL_PTR,
                      "LstmNonMindirCPUKernel malloc weight_project_ptr_ failed.");
    running_buffer_.push_back(weight_project_ptr_);
    PackLstmWeightWithStride(weight_project_ptr_, weight_proj_data, batch, lstm_param_->hidden_size_,
                             lstm_param_->output_size_, lstm_param_->proj_col_align_, lstm_param_->bidirectional_,
                             lstm_param_->hidden_size_ * lstm_param_->output_size_, nullptr);
  } else {
#ifdef ENABLE_AVX
    weight_project_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_size));
    MS_CHECK_TRUE_MSG(weight_project_ptr_ != nullptr, lite::RET_NULL_PTR,
                      "LstmNonMindirCPUKernel malloc weight_project_ptr_ failed.");
    running_buffer_.push_back(weight_project_ptr_);
    for (int i = 0; i < batch; ++i) {
      const float *src_batch = weight_proj_data + i * lstm_param_->hidden_size_ * lstm_param_->output_size_;
      float *dst_batch = weight_project_ptr_ + i * lstm_param_->hidden_size_ * lstm_param_->proj_col_align_;
      RowMajor2Col32Major(src_batch, dst_batch, lstm_param_->output_size_, lstm_param_->hidden_size_);
    }
#else
    weight_project_ptr_ = weight_proj_data;
#endif
  }
  return RET_OK;
}

void LstmMindirFp32CPUKernel::LstmUnidirectional(float *output, const float *weight_h, const float *state_bias,
                                                 float *hidden_state, float *cell_state, const float *weight_project,
                                                 float *intermediate_states, float **buffer, bool is_backward) {
  float *gate = buffer[kInputGateIndex];
  float *input_gate = gate;
  float *forget_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * C2NUM;
  float *cell_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * C3NUM;
  float *output_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_;
  float *tmp = buffer[kTempHiddenOutputIndex];
  int dir_mult = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  for (int t = 0; t < lstm_param_->seq_len_; t++) {
    int real_t = is_backward ? lstm_param_->seq_len_ - t - C1NUM : t;
    float *input_gate_t = input_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *forget_gate_t = forget_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *cell_gate_t = cell_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *output_gate_t = output_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    // Sequence, Batch, DirMul, Hidden
    LstmStepUnit(tmp, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias, weight_project,
                 hidden_state, cell_state, buffer, lstm_param_);
    int seq_offset = real_t * lstm_param_->batch_ * dir_mult * lstm_param_->output_size_;
    for (int b = 0; b < lstm_param_->batch_; b++) {
      int batch_offset = b * dir_mult * lstm_param_->output_size_;
      float *output_ptr = output + seq_offset + batch_offset;
      memcpy(output_ptr, tmp + b * lstm_param_->output_size_, lstm_param_->output_size_ * sizeof(float));
    }
    if (intermediate_states) {
      RecordStates(hidden_state, cell_state, input_gate_t, output_gate_t, forget_gate_t, cell_gate_t,
                   intermediate_states, real_t);
    }
  }
}

void LstmMindirFp32CPUKernel::RecordStates(const float *hidden_state, float *cell_state, float *input_gate,
                                           const float *output_gate, float *forget_gate, const float *cell_gate,
                                           float *intermediate_states, int step) {
  float *states = intermediate_states;
  auto hidden_size = lstm_param_->batch_ * lstm_param_->output_size_;
  auto state_size = lstm_param_->batch_ * lstm_param_->hidden_size_;
  if (state_size < 0) {
    MS_LOG(ERROR) << "state size should be greater than or equal to zero.";
    return;
  }
  auto hidden_stride = step * lstm_param_->output_step_;
  auto hidden_seq_stride = lstm_param_->seq_len_ * lstm_param_->output_step_;
  auto other_output_step = lstm_param_->bidirectional_ ? C2NUM * lstm_param_->batch_ * lstm_param_->hidden_size_
                                                       : lstm_param_->batch_ * lstm_param_->hidden_size_;
  auto stride = step * other_output_step;
  auto seq_stride = lstm_param_->seq_len_ * other_output_step;
  memcpy(states + hidden_stride, hidden_state, hidden_size * sizeof(float));
  stride += hidden_seq_stride;
  memcpy(states + stride, cell_state, state_size * sizeof(float));
  stride += seq_stride;
  memcpy(states + stride, input_gate, state_size * sizeof(float));
  stride += seq_stride;
  memcpy(states + stride, output_gate, state_size * sizeof(float));
  stride += seq_stride;
  memcpy(states + stride, forget_gate, state_size * sizeof(float));
  stride += seq_stride;
  memcpy(states + stride, cell_gate, state_size * sizeof(float));
}
}  // namespace mindspore::kernel
