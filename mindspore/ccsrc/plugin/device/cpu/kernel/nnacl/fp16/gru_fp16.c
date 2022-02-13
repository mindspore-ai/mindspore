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
#include "nnacl/fp16/gru_fp16.h"
#include <string.h>
#include "nnacl/fp16/lstm_fp16.h"
#include "nnacl/fp16/activation_fp16.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"

void GruStepUnitFp16(float16_t *output, float16_t *update_gate, float16_t *reset_gate, float16_t *hidden_buffer,
                     const float16_t *state_weight, const float16_t *state_bias, float16_t *hidden_state,
                     float16_t *buffer[4], const GruParameter *gru_param) {
  float16_t *packed_state = buffer[2];
  float16_t *state_gate = buffer[3];
  bool is_vec = gru_param->batch_ == 1;

  const float16_t *state_update_weight = state_weight;
  const float16_t *state_reset_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_;
  const float16_t *state_hidden_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_ * 2;
  float16_t *state_update_gate = state_gate;
  float16_t *state_reset_gate = state_gate + gru_param->batch_ * gru_param->hidden_size_;
  float16_t *state_hidden_buffer = state_gate + gru_param->batch_ * gru_param->hidden_size_ * 2;
  const float16_t *state_update_bias = state_bias;
  const float16_t *state_reset_bias = state_bias + gru_param->hidden_size_;
  const float16_t *state_hidden_bias = state_bias + gru_param->hidden_size_ * 2;

  // state * weight
  if (is_vec) {
    LstmMatMulFp16(state_reset_gate, hidden_state, state_reset_weight, state_reset_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    LstmMatMulFp16(state_update_gate, hidden_state, state_update_weight, state_update_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    RowMajor2Col16MajorFp16(hidden_state, packed_state, gru_param->batch_, gru_param->hidden_size_, false);
    LstmMatMulFp16(state_reset_gate, packed_state, state_reset_weight, state_reset_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    LstmMatMulFp16(state_update_gate, packed_state, state_update_weight, state_update_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  }
  ElementAddFp16(update_gate, state_update_gate, update_gate, gru_param->batch_ * gru_param->hidden_size_);
  ElementAddFp16(reset_gate, state_update_gate + gru_param->batch_ * gru_param->hidden_size_, reset_gate,
                 gru_param->batch_ * gru_param->hidden_size_);

  // update reset_gate
  SigmoidFp16(reset_gate, reset_gate, gru_param->batch_ * gru_param->hidden_size_);

  // update update_gate
  SigmoidFp16(update_gate, update_gate, gru_param->batch_ * gru_param->hidden_size_);

  ElementMulFp16(hidden_state, reset_gate, reset_gate, gru_param->batch_ * gru_param->hidden_size_);
  if (is_vec) {
    LstmMatMulFp16(state_hidden_buffer, reset_gate, state_hidden_weight, state_hidden_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    RowMajor2Col16MajorFp16(reset_gate, packed_state, gru_param->batch_, gru_param->hidden_size_, false);
    LstmMatMulFp16(state_hidden_buffer, packed_state, state_hidden_weight, state_hidden_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  }
  ElementAddFp16(hidden_buffer, state_hidden_buffer, hidden_buffer, gru_param->batch_ * gru_param->hidden_size_);

  TanhFp16(hidden_buffer, hidden_buffer, gru_param->batch_ * gru_param->hidden_size_);

  ElementMulFp16(update_gate, hidden_state, hidden_state, gru_param->batch_ * gru_param->hidden_size_);

  ArithmeticParameter parameter;
  parameter.in_elements_num0_ = 1;
  parameter.in_elements_num1_ = gru_param->batch_ * gru_param->hidden_size_;
  float16_t one = 1.0f;
  ElementOptSubFp16(&one, update_gate, update_gate, gru_param->batch_ * gru_param->hidden_size_, &parameter);

  ElementMulAccFp16(update_gate, hidden_buffer, hidden_state, gru_param->batch_ * gru_param->hidden_size_);

  memcpy(output, hidden_state, gru_param->batch_ * gru_param->hidden_size_ * sizeof(float16_t));
}

void GruUnidirectionalFp16(float16_t *output, const float16_t *packed_input, const float16_t *weight_g,
                           const float16_t *weight_r, const float16_t *input_bias, const float16_t *state_bias,
                           float16_t *hidden_state, float16_t *buffer[4], const GruParameter *gru_param,
                           bool is_backward) {
  float16_t *gate = buffer[1];
  for (int i = 0; i < 3; i++) {
    const float16_t *weight_loop = weight_g + gru_param->input_size_ * gru_param->input_col_align_ * i;
    const float16_t *bias_loop = input_bias + gru_param->input_col_align_ * i;
    float16_t *gate_loop = gate + gru_param->seq_len_ * gru_param->batch_ * gru_param->hidden_size_ * i;
    MatMulFp16(packed_input, weight_loop, gate_loop, bias_loop, ActType_No, gru_param->input_size_,
               gru_param->seq_len_ * gru_param->batch_, gru_param->hidden_size_, gru_param->hidden_size_, OutType_Nhwc);
  }

  float16_t *update_gate = gate;
  float16_t *reset_gate = gate + gru_param->seq_len_ * gru_param->batch_ * gru_param->hidden_size_;
  float16_t *hidden_buffer = gate + gru_param->seq_len_ * gru_param->batch_ * gru_param->hidden_size_ * 2;
  for (int t = 0; t < gru_param->seq_len_; t++) {
    int real_t = is_backward ? gru_param->seq_len_ - t - 1 : t;
    float16_t *update_gate_t = update_gate + gru_param->batch_ * gru_param->hidden_size_ * real_t;
    float16_t *reset_gate_t = reset_gate + gru_param->batch_ * gru_param->hidden_size_ * real_t;
    float16_t *hidden_buffer_t = hidden_buffer + gru_param->batch_ * gru_param->hidden_size_ * real_t;
    float16_t *output_ptr = output + real_t * gru_param->output_step_;
    GruStepUnitFp16(output_ptr, update_gate_t, reset_gate_t, hidden_buffer_t, weight_r, state_bias, hidden_state,
                    buffer, gru_param);
  }
}

void GruFp16(float16_t *output, const float16_t *input, const float16_t *weight_g, const float16_t *weight_r,
             const float16_t *input_bias, const float16_t *state_bias, float16_t *hidden_state, float16_t *buffer[4],
             int check_seq_len, const GruParameter *gru_param) {
  // forward
  float16_t *packed_input = buffer[0];
  RowMajor2Col16MajorFp16(input, packed_input, gru_param->seq_len_ * gru_param->batch_, gru_param->input_size_, false);
  GruUnidirectionalFp16(output, packed_input, weight_g, weight_r, input_bias, state_bias, hidden_state, buffer,
                        gru_param, false);
  // zero out extra fw outputs
  for (int t = check_seq_len; t < gru_param->seq_len_; t++) {
    float16_t *output_ptr = output + t * gru_param->output_step_;
    for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
      output_ptr[i] = 0.0f;
    }
  }

  // backward
  if (gru_param->bidirectional_) {
    const float16_t *backward_weight_g = weight_g + 3 * gru_param->input_col_align_ * gru_param->input_size_;
    const float16_t *backward_weight_r = weight_r + 3 * gru_param->state_col_align_ * gru_param->hidden_size_;
    const float16_t *backward_input_bias = input_bias + 3 * gru_param->input_col_align_;
    const float16_t *backward_state_bias = state_bias + 3 * gru_param->state_col_align_;
    float16_t *backward_output = output + gru_param->batch_ * gru_param->hidden_size_;
    float16_t *backward_hidden_state = hidden_state + gru_param->batch_ * gru_param->hidden_size_;
    GruUnidirectionalFp16(backward_output, packed_input, backward_weight_g, backward_weight_r, backward_input_bias,
                          backward_state_bias, backward_hidden_state, buffer, gru_param, true);

    // zero out extra bw outputs
    for (int t = gru_param->seq_len_ - 1; t >= check_seq_len; t--) {
      float16_t *output_ptr = backward_output + t * gru_param->output_step_;
      for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
        output_ptr[i] = 0.0f;
      }
    }
  }
}
