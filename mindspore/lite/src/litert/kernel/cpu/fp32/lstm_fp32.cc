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

#include "src/litert/kernel/cpu/fp32/lstm_fp32.h"
#include <cfloat>
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LSTM;

namespace mindspore::kernel {
namespace {
constexpr int kOutputHiddenStatusIndex = 1;
constexpr int kOutputCellStatusIndex = 2;
}  // namespace

int LstmInputMulWeightRun(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<const LstmCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  kernel->InputWeightMatMul(task_id);
  return RET_OK;
}

int LstmSequenceLoopRun(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<LstmCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->DoSequenceLoop(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM: Do Sequence-loop failed.";
  }
  return ret;
}

void LstmCPUKernel::FreeRunBuffer() {
  for (auto data : buffer_running_malloc_) {
    ms_context_->allocator->Free(data);
  }
  buffer_running_malloc_.clear();
}

int LstmCPUKernel::InitInputWeightBias() {
  // malloc and init input * weight right matrix buffer
  // input -- row: seq_len * batch; col: input_size
  // weight -- row: hidden_size; col: input_size, need transpose
  // result -- row: seq_len * batch; col: hidden_size
  weight_i_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(
    weight_batch_ * lstm_param_->input_col_align_ * lstm_param_->input_size_ * sizeof(float)));
  if (weight_i_ptr_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc weight_i_ptr_ error.";
    return RET_ERROR;
  }
  buffer_running_malloc_.push_back(weight_i_ptr_);
  int i_index = (in_tensors_.size() == mindir_input_tensors) ? combined_weights_index : onnx_weight_i_index;
  const int *weights_order = (in_tensors_.size() == mindir_input_tensors) ? weights_order_IFOG : nullptr;
  auto weight_i = in_tensors_.at(i_index);
  auto weight_i_data = reinterpret_cast<float *>(weight_i->data());

  CHECK_NULL_RETURN(weight_i_data);
  int cw_size = (lstm_param_->input_size_ * lstm_param_->hidden_size_);
  int hh_size = (lstm_param_->hidden_size_ * lstm_param_->hidden_size_);
  int b_size = (lstm_param_->hidden_size_);
  bool has_bias = (weight_batch_ * (cw_size + hh_size) < weight_i->ElementsNum()) ? true : false;
  int stride = (gpu_orig_state_) ? gate_num * (cw_size + hh_size) : gate_num * (cw_size);
  PackLstmWeightWithStride(weight_i_ptr_, weight_i_data, weight_batch_, lstm_param_->input_size_,
                           lstm_param_->hidden_size_, lstm_param_->input_col_align_, lstm_param_->bidirectional_,
                           stride, weights_order);
  // input bias
  input_bias_ = reinterpret_cast<float *>(
    ms_context_->allocator->Malloc(weight_batch_ * lstm_param_->input_col_align_ * sizeof(float)));
  if (input_bias_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc input_bias_ error.";
    return RET_ERROR;
  }
  memset(input_bias_, 0, weight_batch_ * lstm_param_->input_col_align_ * sizeof(float));
  buffer_running_malloc_.push_back(input_bias_);

  int offset = weight_batch_ * (cw_size + hh_size);
  float *bias_data = (has_bias) ? weight_i_data + offset : nullptr;
  int dir_mul = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  int b_stride = (gpu_orig_state_) ? gate_num * (dir_mul * b_size) : gate_num * (b_size);
  if (in_tensors_.size() > mindir_input_tensors) {
    bias_data = reinterpret_cast<float *>(in_tensors_.at(onnx_bias_index)->data());
    PackLstmBias(input_bias_, bias_data, weight_batch_, lstm_param_->hidden_size_, lstm_param_->input_col_align_,
                 lstm_param_->bidirectional_, weights_order);
  } else {
    if (bias_data != nullptr) {
      PackLstmBiasWithStride(input_bias_, bias_data, weight_batch_, lstm_param_->hidden_size_,
                             lstm_param_->input_col_align_, lstm_param_->bidirectional_, b_stride, weights_order);
    }
  }
  return RET_OK;
}

int LstmCPUKernel::InitStateWeightBias() {
  // malloc and init state * weight right matrix buffer, state * weight will be executed seq_len_ times.
  // state -- row: batch; col: hidden_size
  // weight -- row: hidden_size; col: hidden_size, need transpose
  // result -- row: batch; col: hidden_size
  int weight_i_size = weight_batch_ * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  int h_index = (in_tensors_.size() == mindir_input_tensors) ? combined_weights_index : onnx_weight_h_index;
  auto weight_h = in_tensors_.at(h_index);
  auto weight_h_data = (reinterpret_cast<float *>(weight_h->data()));

  int cw_size = (lstm_param_->input_size_ * lstm_param_->hidden_size_);
  int hh_size = (lstm_param_->hidden_size_ * lstm_param_->hidden_size_);
  int b_size = (lstm_param_->hidden_size_);
  int stride = (gpu_orig_state_) ? gate_num * (cw_size + hh_size) : gate_num * (hh_size);

  if (in_tensors_.size() == mindir_input_tensors) {
    if (gpu_orig_state_) {
      weight_h_data += gate_num * cw_size;
    } else {
      weight_h_data += weight_i_size;
    }
  }
  CHECK_NULL_RETURN(weight_h_data);
  if (!state_is_vec_) {
    weight_h_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(
      weight_batch_ * lstm_param_->state_col_align_ * lstm_param_->hidden_size_ * sizeof(float)));
    if (weight_h_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc weight_h_ptr_ error.";
      return RET_ERROR;
    }
    buffer_running_malloc_.push_back(weight_h_ptr_);
    const int *weights_order = (in_tensors_.size() == mindir_input_tensors) ? weights_order_IFOG : nullptr;
    PackLstmWeightWithStride(weight_h_ptr_, weight_h_data, weight_batch_, lstm_param_->hidden_size_,
                             lstm_param_->hidden_size_, lstm_param_->state_col_align_, lstm_param_->bidirectional_,
                             stride, weights_order);
  } else {
#ifdef ENABLE_AVX
    weight_h_ptr_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(
      weight_batch_ * lstm_param_->state_col_align_ * lstm_param_->hidden_size_ * sizeof(float)));
    if (weight_h_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc weight_h_ptr_ error.";
      return RET_ERROR;
    }
    buffer_running_malloc_.push_back(weight_h_ptr_);
    for (int i = 0; i < weight_batch_; i++) {
      const float *src_batch = weight_h_data + i * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
      float *dst_batch = weight_h_ptr_ + i * lstm_param_->state_col_align_ * lstm_param_->hidden_size_;
      RowMajor2Col32Major(src_batch, dst_batch, lstm_param_->hidden_size_, lstm_param_->hidden_size_);
    }
#else
    weight_h_ptr_ = weight_h_data;
#endif
  }

  // state bias
  int weight_h_size = weight_batch_ * lstm_param_->hidden_size_ * lstm_param_->hidden_size_;
  int bias_size = weight_batch_ * lstm_param_->hidden_size_;
  state_bias_ = reinterpret_cast<float *>(
    ms_context_->allocator->Malloc(weight_batch_ * lstm_param_->state_col_align_ * sizeof(float)));
  if (state_bias_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc state_bias_ error.";
    return RET_ERROR;
  }
  memset(state_bias_, 0, weight_batch_ * lstm_param_->state_col_align_ * sizeof(float));
  buffer_running_malloc_.push_back(state_bias_);
  // if ONNX, secend bias is also present order IOFG
  if (in_tensors_.size() > mindir_input_tensors) {
    float *state_bias =
      reinterpret_cast<float *>(in_tensors_.at(onnx_bias_index)->data()) + gate_num * lstm_param_->hidden_size_;
    CHECK_NULL_RETURN(state_bias);
    PackLstmBias(state_bias_, state_bias, weight_batch_, lstm_param_->hidden_size_, lstm_param_->state_col_align_,
                 lstm_param_->bidirectional_, nullptr);
  } else if (weight_h->ElementsNum() - weight_i_size - weight_h_size - C2NUM * bias_size == 0) {
    // mindir from device "GPU", secend bias is also present order IFOG
    int dir_mul = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
    int bias_offset = (gpu_orig_state_) ? gate_num * ((dir_mul - C1NUM) * cw_size + dir_mul * hh_size + b_size)
                                        : weight_h_size + bias_size;
    float *state_bias = weight_h_data + bias_offset;
    int b_stride = (gpu_orig_state_) ? gate_num * (b_size * C2NUM) : gate_num * b_size;
    PackLstmBiasWithStride(state_bias_, state_bias, weight_batch_, lstm_param_->hidden_size_,
                           lstm_param_->state_col_align_, lstm_param_->bidirectional_, b_stride, weights_order_IFOG);
  }
  return RET_OK;
}

int LstmCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  std::vector<int> in_shape = input->shape();
  lstm_param_->seq_len_ = in_shape.at(FIRST_INPUT);
  lstm_param_->batch_ = in_shape.at(SECOND_INPUT);
  lstm_param_->input_size_ = in_shape.at(THIRD_INPUT);

  auto weight_i = in_tensors_.at(onnx_weight_i_index);
  std::vector<int> w_shape = weight_i->shape();
  if (in_tensors_.size() == mindir_input_tensors) {
    hidden_state_input_index_ = mindir_hidden_state_input_index;
    cell_state_input_index_ = mindir_cell_state_input_index;
    lstm_param_->hidden_size_ = w_shape.at(THIRD_INPUT);
  } else {
    lstm_param_->hidden_size_ = w_shape.at(SECOND_INPUT) / gate_num;
  }
  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? C2NUM * lstm_param_->batch_ * lstm_param_->hidden_size_
                                                          : lstm_param_->batch_ * lstm_param_->hidden_size_;
  weight_batch_ = lstm_param_->bidirectional_ ? C2NUM * gate_num : gate_num;
  state_is_vec_ = lstm_param_->batch_ == 1;
  // determine FB origin
  gpu_orig_state_ = false;
  if (in_tensors_.size() == mindir_input_tensors) {
    gpu_orig_state_ = gpu_orig_cfg_;
    auto weight_t = in_tensors_.at(combined_weights_index);
    int cw_size = (lstm_param_->input_size_ * lstm_param_->hidden_size_);
    int hh_size = (lstm_param_->hidden_size_ * lstm_param_->hidden_size_);
    int b_size = (lstm_param_->hidden_size_);
    bool has_bias = (weight_batch_ * (cw_size + hh_size) < weight_t->ElementsNum()) ? true : false;
    // if bias exist we can determine the gpu_orig_state_
    if (has_bias) {
      gpu_orig_state_ =
        (weight_batch_ * (cw_size + hh_size + C2NUM * b_size) == weight_t->ElementsNum()) ? true : false;
    }
  }

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
  input_thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(lstm_param_->input_col_align_, col_tile_));
  MS_CHECK_FALSE(input_thread_count_ == 0, RET_ERROR);
  input_thread_stride_ = UP_DIV(UP_DIV(lstm_param_->input_col_align_, col_tile_), input_thread_count_);

  state_row_tile_ = row_tile_;
  state_col_tile_ = col_tile_;
#ifdef ENABLE_AVX
  if (state_is_vec_) {
    state_row_tile_ = 1;
    state_col_tile_ = C8NUM;
  }
#endif

  lstm_param_->state_row_align_ = state_is_vec_ ? 1 : UP_ROUND(lstm_param_->batch_, state_row_tile_);
#ifdef ENABLE_AVX
  lstm_param_->state_col_align_ = UP_ROUND(lstm_param_->hidden_size_, state_col_tile_);
#else
  lstm_param_->state_col_align_ =
    state_is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, state_col_tile_);
#endif
  return RET_OK;
}

int LstmCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), mindir_input_tensors);
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    CHECK_NULL_RETURN(in_tensors_.at(i));
  }
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_3D);
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    CHECK_NULL_RETURN(out_tensors_.at(i));
  }
  CHECK_NULL_RETURN(lstm_param_);
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

  return RET_OK;
}

int LstmCPUKernel::MallocRunBuffer(bool is_double) {
  size_t whole_size = 0;
  std::vector<size_t> segments;
  int scale = is_double ? C2NUM : 1;
  size_t segment = gate_num * lstm_param_->seq_len_ * lstm_param_->batch_ *
                   lstm_param_->hidden_size_;  // input * weight for result matrix
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = 0;
  if (!state_is_vec_) {
    segment = lstm_param_->state_row_align_ * lstm_param_->hidden_size_;  // state * weight for left matirx
  }
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = gate_num * lstm_param_->batch_ * lstm_param_->hidden_size_;  // state gate buffer
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = 0;
  if (!(lstm_param_->zoneout_cell_ >= -FLT_EPSILON && lstm_param_->zoneout_cell_ <= FLT_EPSILON)) {
    segment = lstm_param_->batch_ * lstm_param_->hidden_size_;  // state_buffer for cell
  }
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = 0;
  if (!(lstm_param_->zoneout_hidden_ >= -FLT_EPSILON && lstm_param_->zoneout_hidden_ <= FLT_EPSILON)) {
    segment = lstm_param_->batch_ * lstm_param_->hidden_size_;  // state_buffer for hidden
  }
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = 0;
#ifdef ENABLE_AVX
  if (state_is_vec_) {  // vec matmul need to malloc dst
    bool output_need_packed = lstm_param_->hidden_size_ % state_col_tile_;
    if (output_need_packed) {
      int out_channel = lstm_param_->hidden_size_;
      int oc_block_num = UP_DIV(out_channel, state_col_tile_);
      MS_ASSERT(ms_context_->allocator != nullptr);
      segment = lstm_param_->batch_ * oc_block_num * state_col_tile_;  //  tmp output data
    }
  }
#endif
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = 0;
  if (!(in_tensors_.size() > mindir_input_tensors)) {
    segment = lstm_param_->batch_ * lstm_param_->hidden_size_;
  }
  segments.push_back(segment);
  whole_size += segment * scale;

  segment =
    lstm_param_->input_row_align_ * lstm_param_->input_size_;  // input * weight for left matrix, which only once
  whole_size += segment;

  auto whole_memory = reinterpret_cast<float *>(ms_context_->allocator->Malloc(whole_size * sizeof(float)));
  if (whole_memory == nullptr) {
    MS_LOG(ERROR) << "LSTM: malloc " << whole_size << " bytes for running failed.";
    return RET_ERROR;
  }
  buffer_running_malloc_.push_back(whole_memory);
  MS_ASSERT(segments.size() == C7NUM);
  for (int i = 0; i < C7NUM; i++) {
    buffer_forward_[i] = nullptr;
    if (segments[i] == 0) {
      continue;
    }
    buffer_forward_[i] = whole_memory;
    whole_memory += segments[i];
  }
  if (is_double) {
    for (int i = 0; i < C7NUM; i++) {
      buffer_backward_[i] = nullptr;
      if (segments[i] == 0) {
        continue;
      }
      buffer_backward_[i] = whole_memory;
      whole_memory += segments[i];
    }
  }
  packed_input_ = whole_memory;
  return RET_OK;
}

void LstmCPUKernel::InputWeightMatMul(int task_id) const {
  int current_start_oc = task_id * input_thread_stride_ * col_tile_;
  int current_rest_oc = 0;
  current_rest_oc = lstm_param_->hidden_size_ - current_start_oc;
  int cur_oc = MSMIN(input_thread_stride_ * col_tile_, current_rest_oc);
  if (cur_oc <= 0) {
    return;
  }

  auto b = weight_loop_ + current_start_oc * lstm_param_->input_size_;
  auto c = gate_loop_ + current_start_oc;
  auto bias = (bias_loop_ == nullptr) ? nullptr : bias_loop_ + current_start_oc;
  MatMulOpt(packed_input_, b, c, bias, ActType_No, lstm_param_->input_size_,
            lstm_param_->seq_len_ * lstm_param_->batch_, cur_oc, lstm_param_->hidden_size_, OutType_Nhwc);
}

int LstmCPUKernel::DoSequenceLoop(int task_id) {
  if (task_id == 0) {
    LstmForwardLoop(buffer_forward_);
    return RET_OK;
  }
  if (task_id == 1) {
    LstmBackwardLoop(buffer_backward_);
    return RET_OK;
  }
  return RET_ERROR;
}

int LstmCPUKernel::LstmPreProcessWithInput(const float *weight_i, const float *input_bias, float *dst) {
  for (int i = 0; i < gate_num; i++) {
    weight_loop_ = weight_i + lstm_param_->input_size_ * lstm_param_->input_col_align_ * i;
    bias_loop_ = input_bias + lstm_param_->input_col_align_ * i;
    gate_loop_ = dst + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * i;
    auto ret = ParallelLaunch(this->ms_context_, LstmInputMulWeightRun, this, input_thread_count_);
    if (ret != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void LstmCPUKernel::LstmUnidirectional(float *output, const float *weight_h, const float *state_bias,
                                       float *hidden_state, float *cell_state, float *intermediate_states,
                                       float *buffer[], bool is_backward) {
  float *gate = buffer[input_gate_index];
  float *input_gate = gate;
  float *forget_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * C2NUM;
  float *cell_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * C3NUM;
  float *output_gate = gate + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_;
  float *tmp = buffer[tmp_hidden_output_index];
  int dir_mult = lstm_param_->bidirectional_ ? C2NUM : C1NUM;
  for (int t = 0; t < lstm_param_->seq_len_; t++) {
    int real_t = is_backward ? lstm_param_->seq_len_ - t - C1NUM : t;
    float *input_gate_t = input_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *forget_gate_t = forget_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *cell_gate_t = cell_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    float *output_gate_t = output_gate + lstm_param_->batch_ * lstm_param_->hidden_size_ * real_t;
    // if ONNX
    if (in_tensors_.size() > mindir_input_tensors) {
      // Sequence, DirMul, Batch, Hidden
      float *output_ptr = output + real_t * lstm_param_->output_step_;

      LstmStepUnit(output_ptr, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias,
                   hidden_state, cell_state, buffer, lstm_param_);
    } else {
      // Sequence, Batch, DirMul, Hidden
      LstmStepUnit(tmp, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias, hidden_state,
                   cell_state, buffer, lstm_param_);
      int seq_offset = real_t * lstm_param_->batch_ * dir_mult * lstm_param_->hidden_size_;
      for (int b = 0; b < lstm_param_->batch_; b++) {
        int batch_offset = b * dir_mult * lstm_param_->hidden_size_;
        float *output_ptr = output + seq_offset + batch_offset;
        memcpy(output_ptr, tmp + b * lstm_param_->hidden_size_, lstm_param_->hidden_size_ * sizeof(float));
      }
    }
    if (intermediate_states) {
      RecordStates(hidden_state, cell_state, input_gate_t, output_gate_t, forget_gate_t, cell_gate_t,
                   intermediate_states, real_t);
    }
  }
}

void LstmCPUKernel::RecordStates(const float *hidden_state, float *cell_state, float *input_gate,
                                 const float *output_gate, float *forget_gate, const float *cell_gate,
                                 float *intermediate_states, int step) {
  float *states = intermediate_states;
  auto state_size = lstm_param_->batch_ * lstm_param_->hidden_size_;
  if (state_size < 0) {
    MS_LOG(ERROR) << "state size should be greater than or equal to zero.";
    return;
  }
  auto stride = step * lstm_param_->output_step_;
  auto seq_stride = lstm_param_->seq_len_ * lstm_param_->output_step_;
  memcpy(states + stride, hidden_state, state_size * sizeof(float));
  stride += seq_stride;
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

void LstmCPUKernel::LstmForwardLoop(float *buffer[]) {
  auto *output = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  auto *hidden_state = reinterpret_cast<float *>(out_tensors_.at(1)->data());
  auto *cell_state = reinterpret_cast<float *>(out_tensors_.at(C2NUM)->data());
  LstmUnidirectional(output, weight_h_ptr_, state_bias_, hidden_state, cell_state, intermediate_states_, buffer, false);
}

void LstmCPUKernel::LstmBackwardLoop(float *buffer[]) {
  auto *output = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  auto *hidden_state = reinterpret_cast<float *>(out_tensors_.at(1)->data());
  auto *cell_state = reinterpret_cast<float *>(out_tensors_.at(C2NUM)->data());
  const float *backward_weight_h = weight_h_ptr_ + gate_num * lstm_param_->state_col_align_ * lstm_param_->hidden_size_;
  const float *backward_state_bias = state_bias_ + gate_num * lstm_param_->state_col_align_;
  float *backward_output = output + lstm_param_->batch_ * lstm_param_->hidden_size_;
  if (in_tensors_.size() == mindir_input_tensors) {
    backward_output = output + lstm_param_->hidden_size_;
  }
  float *backward_cell_state = cell_state + lstm_param_->batch_ * lstm_param_->hidden_size_;
  float *backward_hidden_state = hidden_state + lstm_param_->batch_ * lstm_param_->hidden_size_;
  float *intermediate_states = nullptr;
  if (intermediate_states_) {
    intermediate_states = intermediate_states_ + lstm_param_->batch_ * lstm_param_->hidden_size_;
  }
  LstmUnidirectional(backward_output, backward_weight_h, backward_state_bias, backward_hidden_state,
                     backward_cell_state, intermediate_states, buffer, true);
}

int LstmCPUKernel::ExecuteUnidirectionalOrSingleThread() {
  auto ret = LstmPreProcessWithInput(weight_i_ptr_, input_bias_, buffer_forward_[input_gate_index]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM Forward: Input-MatMul running failed.";
    return RET_ERROR;
  }
  LstmForwardLoop(buffer_forward_);

  // backward
  if (lstm_param_->bidirectional_) {
    const float *backward_weight_i =
      weight_i_ptr_ + gate_num * lstm_param_->input_col_align_ * lstm_param_->input_size_;
    const float *backward_input_bias = input_bias_ + gate_num * lstm_param_->input_col_align_;
    ret = LstmPreProcessWithInput(backward_weight_i, backward_input_bias, buffer_forward_[input_gate_index]);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "LSTM Backward: Input-MatMul running failed.";
      return RET_ERROR;
    }
    LstmBackwardLoop(buffer_forward_);
  }
  return RET_OK;
}

int LstmCPUKernel::ExecuteBidirectionalWithMultiThread() {
  auto ret = LstmPreProcessWithInput(weight_i_ptr_, input_bias_, buffer_forward_[input_gate_index]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM Forward: Input-MatMul running failed.";
    return RET_ERROR;
  }
  const float *backward_weight_i = weight_i_ptr_ + gate_num * lstm_param_->input_col_align_ * lstm_param_->input_size_;
  const float *backward_input_bias = input_bias_ + gate_num * lstm_param_->input_col_align_;
  ret = LstmPreProcessWithInput(backward_weight_i, backward_input_bias, buffer_backward_[input_gate_index]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM Backward: Input-MatMul running failed.";
    return RET_ERROR;
  }
  ret = ParallelLaunch(this->ms_context_, LstmSequenceLoopRun, this, C2NUM);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM: Do sequence-loop failed.";
  }
  return ret;
}

int LstmCPUKernel::Run() {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);
  auto input_ptr = reinterpret_cast<float *>(input->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<float *>(output->data());
  CHECK_NULL_RETURN(output_ptr);

  auto hidden_state = in_tensors_.at(hidden_state_input_index_);
  CHECK_NULL_RETURN(hidden_state->data());
  auto cell_state = in_tensors_.at(cell_state_input_index_);
  CHECK_NULL_RETURN(cell_state->data());

  auto output_hidden_state = out_tensors_[kOutputHiddenStatusIndex];
  CHECK_NULL_RETURN(output_hidden_state->data());
  (void)memcpy(output_hidden_state->data(), hidden_state->data(), hidden_state->ElementsNum() * sizeof(float));
  auto output_cell_state = out_tensors_[kOutputCellStatusIndex];
  CHECK_NULL_RETURN(output_cell_state->data());
  (void)memcpy(output_cell_state->data(), cell_state->data(), cell_state->ElementsNum() * sizeof(float));

  auto ret = InitInputWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitInputWeightBias error.";
    FreeRunBuffer();
    return RET_ERROR;
  }

  ret = InitStateWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitStateWeightBias error.";
    FreeRunBuffer();
    return RET_ERROR;
  }

  bool is_bidirectional_with_multi_thread = thread_num_ != 1 && lstm_param_->bidirectional_;
  ret = MallocRunBuffer(is_bidirectional_with_multi_thread);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel MallocRunBuffer Error.";
    FreeRunBuffer();
    return RET_ERROR;
  }

  PackLstmInput(input_ptr, packed_input_, lstm_param_->seq_len_ * lstm_param_->batch_, lstm_param_->input_size_);
  if (IsTrain() && IsTrainable()) {
    intermediate_states_ = reinterpret_cast<float *>(out_tensors_[out_intermediate_states_index]->data());
  }
  CHECK_NULL_RETURN(weight_h_ptr_);
  CHECK_NULL_RETURN(weight_i_ptr_);
  CHECK_NULL_RETURN(input_bias_);
  CHECK_NULL_RETURN(state_bias_);
  if (is_bidirectional_with_multi_thread) {
    ret = ExecuteBidirectionalWithMultiThread();
  } else {
    ret = ExecuteUnidirectionalOrSingleThread();
  }
  FreeRunBuffer();
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LSTM, LiteKernelCreator<LstmCPUKernel>)
}  // namespace mindspore::kernel
