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

#include "src/litert/kernel/cpu/fp32/lstm_fp32_base.h"
#include <vector>
#include "include/errorcode.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr size_t kMindirInputTensorNum = 4;
constexpr int kGateNum = 4;
constexpr int kOutIntermediateStatesIndex = 3;
constexpr int kInputGateIndex = 0;
}  // namespace

int LstmSequenceLoopRun(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<LstmFp32BaseCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->DoSequenceLoop(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM: Do Sequence-loop failed.";
  }
  return ret;
}

int LstmFp32BaseCPUKernel::Prepare() {
  MS_CHECK_TRUE_MSG(in_tensors_.size() == kMindirInputTensorNum || in_tensors_.size() >= C6NUM,
                    lite::RET_INPUT_TENSOR_ERROR, "Lstm's input-num is invalid.");
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

int LstmFp32BaseCPUKernel::ReSize() {
  auto input = in_tensors_.front();
  std::vector<int> in_shape = input->shape();
  MS_CHECK_TRUE_MSG(in_shape.size() == C3NUM, lite::RET_INPUT_TENSOR_ERROR,
                    "The dims of LSTM's first input must be 3.");
  lstm_param_->seq_len_ = in_shape.at(FIRST_INPUT);
  lstm_param_->batch_ = in_shape.at(SECOND_INPUT);
  lstm_param_->input_size_ = in_shape.at(THIRD_INPUT);

  auto h_init_shape = in_tensors_.at(hidden_init_index_)->shape();
  auto c_init_shape = in_tensors_.at(cell_init_index_)->shape();
  lstm_param_->hidden_size_ = c_init_shape.back();
  lstm_param_->output_size_ = h_init_shape.back();

  lstm_param_->output_step_ = lstm_param_->bidirectional_ ? C2NUM * lstm_param_->batch_ * lstm_param_->output_size_
                                                          : lstm_param_->batch_ * lstm_param_->output_size_;
  weight_segment_num_ = lstm_param_->bidirectional_ ? C2NUM * kGateNum : kGateNum;

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

  state_row_tile_ = row_tile_;
  state_col_tile_ = col_tile_;
#ifdef ENABLE_AVX
  if (lstm_param_->batch_ == 1) {
    state_row_tile_ = 1;
    state_col_tile_ = C8NUM;
  }
#endif

  lstm_param_->state_row_align_ = lstm_param_->batch_ == 1 ? 1 : UP_ROUND(lstm_param_->batch_, state_row_tile_);
#ifdef ENABLE_AVX
  lstm_param_->state_col_align_ = UP_ROUND(lstm_param_->hidden_size_, state_col_tile_);
  lstm_param_->proj_col_align_ = UP_ROUND(lstm_param_->output_size_, state_col_tile_);
#else
  lstm_param_->state_col_align_ =
    lstm_param_->batch_ == 1 ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, state_col_tile_);
  lstm_param_->proj_col_align_ =
    lstm_param_->batch_ == 1 ? lstm_param_->output_size_ : UP_ROUND(lstm_param_->output_size_, state_col_tile_);
#endif
  return RET_OK;
}

int LstmFp32BaseCPUKernel::Run() {
  auto input = in_tensors_.at(FIRST_INPUT);
  auto output = out_tensors_.at(FIRST_INPUT);
  auto input_ptr = reinterpret_cast<float *>(input->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<float *>(output->data());
  CHECK_NULL_RETURN(output_ptr);

  auto hidden_state = in_tensors_.at(hidden_init_index_);
  CHECK_NULL_RETURN(hidden_state->data());
  auto cell_state = in_tensors_.at(cell_init_index_);
  CHECK_NULL_RETURN(cell_state->data());

  auto output_hidden_state = out_tensors_[SECOND_INPUT];
  CHECK_NULL_RETURN(output_hidden_state->data());
  (void)memcpy(output_hidden_state->data(), hidden_state->data(), hidden_state->ElementsNum() * sizeof(float));
  auto output_cell_state = out_tensors_[THIRD_INPUT];
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

  ret = InitProjectWeight();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmCPUKernel InitProjectWeight error.";
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
    intermediate_states_ = reinterpret_cast<float *>(out_tensors_[kOutIntermediateStatesIndex]->data());
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

void LstmFp32BaseCPUKernel::FreeRunBuffer() {
  for (auto data : running_buffer_) {
    ms_context_->allocator->Free(data);
  }
  running_buffer_.clear();
}

int LstmFp32BaseCPUKernel::MallocRunBuffer(bool is_double) {
  bool need_zone = lstm_param_->zoneout_cell_ < -FLT_EPSILON || lstm_param_->zoneout_cell_ > FLT_EPSILON;
  size_t whole_size = 0;
  std::vector<size_t> segments;
  int scale = is_double ? C2NUM : 1;
  size_t segment = kGateNum * lstm_param_->seq_len_ * lstm_param_->batch_ *
                   lstm_param_->hidden_size_;  // 0: input * weight for result matrix
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = lstm_param_->batch_ == 1
              ? 0
              : lstm_param_->state_row_align_ * lstm_param_->output_size_;  // 1: state * weight for left matirx
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = kGateNum * lstm_param_->batch_ * lstm_param_->hidden_size_;  // 2: state gate buffer
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = need_zone ? lstm_param_->batch_ * lstm_param_->hidden_size_ : 0;  // 3: state_buffer for cell
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = need_zone ? lstm_param_->batch_ * lstm_param_->output_size_ : 0;  // 4: state_buffer for hidden
  segments.push_back(segment);
  whole_size += segment * scale;

  segment = 0;
#ifdef ENABLE_AVX
  bool output_need_packed = lstm_param_->hidden_size_ % state_col_tile_;
  if (lstm_param_->batch_ == 1 && output_need_packed) {  // vec matmul need to malloc dst
    int out_channel = lstm_param_->hidden_size_;
    int oc_block_num = UP_DIV(out_channel, state_col_tile_);
    MS_ASSERT(ms_context_->allocator != nullptr);
    segment = lstm_param_->batch_ * oc_block_num * state_col_tile_;  // 5: tmp output data
  }
#endif
  segments.push_back(segment);
  whole_size += segment * scale;

  if (in_tensors_.size() == C7NUM || lstm_param_->project_size_ != 0) {
    segment = lstm_param_->batch_ == 1 ? 0 : lstm_param_->state_row_align_ * lstm_param_->hidden_size_ * scale;
    segments.push_back(segment);  // 6: project-layer input
    whole_size += segment;
    segment = 0;
#ifdef ENABLE_AVX
    segment =
      output_need_packed ? lstm_param_->batch_ * UP_ROUND(lstm_param_->output_size_, state_col_tile_) * scale : 0;
#endif
    segments.push_back(segment);  // 7: project-layer output
    whole_size += segment;
  } else {
    (void)segments.insert(segments.end(), C2NUM, 0);
  }

  segment = 0;
  if (in_tensors_.size() == kMindirInputTensorNum) {
    segment = lstm_param_->batch_ * lstm_param_->output_size_;
  }
  segments.push_back(segment);
  whole_size += segment * scale;

  segment =
    lstm_param_->input_row_align_ * lstm_param_->input_size_;  // input * weight for left matrix, which only once
  whole_size += segment;

  auto whole_memory = reinterpret_cast<float *>(ms_context_->allocator->Malloc(whole_size * sizeof(float)));
  MS_CHECK_TRUE_MSG(whole_memory != nullptr, RET_ERROR, "LSTM: malloc failed.");
  running_buffer_.push_back(whole_memory);
  MS_ASSERT(segments.size() == C9NUM);
  auto Allocate = [&whole_memory, &segments](float **buffer) mutable {
    for (int i = 0; i < C9NUM; ++i) {
      buffer[i] = nullptr;
      if (segments[i] == 0) {
        continue;
      }
      buffer[i] = whole_memory;
      whole_memory += segments[i];
    }
  };
  Allocate(buffer_forward_);
  if (is_double) {
    Allocate(buffer_backward_);
  }
  packed_input_ = whole_memory;
  return RET_OK;
}

int LstmFp32BaseCPUKernel::ExecuteBidirectionalWithMultiThread() {
  auto ret = LstmPreProcessWithInput(weight_i_ptr_, input_bias_, buffer_forward_[kInputGateIndex]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM Forward: Input-MatMul running failed.";
    return RET_ERROR;
  }
  const float *backward_weight_i = weight_i_ptr_ + kGateNum * lstm_param_->input_col_align_ * lstm_param_->input_size_;
  const float *backward_input_bias = input_bias_ + kGateNum * lstm_param_->input_col_align_;
  ret = LstmPreProcessWithInput(backward_weight_i, backward_input_bias, buffer_backward_[kInputGateIndex]);
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

int LstmFp32BaseCPUKernel::ExecuteUnidirectionalOrSingleThread() {
  auto ret = LstmPreProcessWithInput(weight_i_ptr_, input_bias_, buffer_forward_[kInputGateIndex]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LSTM Forward: Input-MatMul running failed.";
    return RET_ERROR;
  }
  LstmForwardLoop(buffer_forward_);

  // backward
  if (lstm_param_->bidirectional_) {
    const float *backward_weight_i =
      weight_i_ptr_ + kGateNum * lstm_param_->input_col_align_ * lstm_param_->input_size_;
    const float *backward_input_bias = input_bias_ + kGateNum * lstm_param_->input_col_align_;
    ret = LstmPreProcessWithInput(backward_weight_i, backward_input_bias, buffer_forward_[kInputGateIndex]);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "LSTM Backward: Input-MatMul running failed.";
      return RET_ERROR;
    }
    LstmBackwardLoop(buffer_forward_);
  }
  return RET_OK;
}

int LstmFp32BaseCPUKernel::LstmPreProcessWithInput(const float *weight_i, const float *input_bias, float *dst) {
  const float *weight{nullptr};
  const float *bias{nullptr};
  float *gate{nullptr};
  int thread_num = MSMIN(op_parameter_->thread_num_, UP_DIV(lstm_param_->input_col_align_, col_tile_));
  MS_CHECK_FALSE(thread_num == 0, RET_ERROR);
  int stride = UP_DIV(UP_DIV(lstm_param_->input_col_align_, col_tile_), thread_num);
  auto MatMulCoreFunc = [this, &weight, &bias, &gate, &stride](void *, int task_id, float, float) {
    int current_start_oc = task_id * stride * col_tile_;
    int current_rest_oc = 0;
    current_rest_oc = lstm_param_->hidden_size_ - current_start_oc;
    int cur_oc = MSMIN(stride * col_tile_, current_rest_oc);
    if (cur_oc <= 0) {
      return RET_OK;
    }

    auto b = weight + current_start_oc * lstm_param_->input_size_;
    auto c = gate + current_start_oc;
    auto bias_ = (bias == nullptr) ? nullptr : bias + current_start_oc;
    MatMulOpt(packed_input_, b, c, bias_, ActType_No, lstm_param_->input_size_,
              lstm_param_->seq_len_ * lstm_param_->batch_, cur_oc, lstm_param_->hidden_size_, OutType_Nhwc);
    return RET_OK;
  };
  for (int i = 0; i < kGateNum; i++) {
    weight = weight_i + lstm_param_->input_size_ * lstm_param_->input_col_align_ * i;
    bias = input_bias + lstm_param_->input_col_align_ * i;
    gate = dst + lstm_param_->seq_len_ * lstm_param_->batch_ * lstm_param_->hidden_size_ * i;
    auto ret = ParallelLaunch(this->ms_context_, MatMulCoreFunc, nullptr, thread_num);
    if (ret != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LstmFp32BaseCPUKernel::DoSequenceLoop(int task_id) {
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

void LstmFp32BaseCPUKernel::LstmForwardLoop(float *buffer[]) {
  auto *output = reinterpret_cast<float *>(out_tensors_.at(FIRST_INPUT)->data());
  auto *hidden_state = reinterpret_cast<float *>(out_tensors_.at(SECOND_INPUT)->data());
  auto *cell_state = reinterpret_cast<float *>(out_tensors_.at(THIRD_INPUT)->data());
  LstmUnidirectional(output, weight_h_ptr_, state_bias_, hidden_state, cell_state, weight_project_ptr_,
                     intermediate_states_, buffer, false);
}

void LstmFp32BaseCPUKernel::LstmBackwardLoop(float *buffer[]) {
  auto *output = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  auto *hidden_state = reinterpret_cast<float *>(out_tensors_.at(1)->data());
  auto *cell_state = reinterpret_cast<float *>(out_tensors_.at(C2NUM)->data());
  const float *backward_weight_h = weight_h_ptr_ + kGateNum * lstm_param_->state_col_align_ * lstm_param_->output_size_;
  const float *backward_state_bias = state_bias_ + kGateNum * lstm_param_->state_col_align_;
  float *backward_output = output + lstm_param_->batch_ * lstm_param_->output_size_;
  if (in_tensors_.size() == kMindirInputTensorNum) {
    backward_output = output + lstm_param_->output_size_;
  }
  float *backward_cell_state = cell_state + lstm_param_->batch_ * lstm_param_->hidden_size_;
  float *backward_hidden_state = hidden_state + lstm_param_->batch_ * lstm_param_->output_size_;
  float *intermediate_states = nullptr;
  if (intermediate_states_) {
    intermediate_states = intermediate_states_ + lstm_param_->batch_ * lstm_param_->output_size_;
  }
  float *backward_weight_project =
    weight_project_ptr_ ? weight_project_ptr_ + lstm_param_->hidden_size_ * lstm_param_->proj_col_align_ : nullptr;
  LstmUnidirectional(backward_output, backward_weight_h, backward_state_bias, backward_hidden_state,
                     backward_cell_state, backward_weight_project, intermediate_states, buffer, true);
}
}  // namespace mindspore::kernel
