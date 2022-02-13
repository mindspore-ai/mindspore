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
#include "nnacl/fp32/gru_fp32.h"
#include <string.h>
#include "nnacl/fp32/lstm_fp32.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
void GruMatMul(float *c, const float *a, const float *b, const float *bias, int row, int deep, int col, bool is_vec) {
  if (is_vec) {
    MatVecMulFp32(a, b, c, bias, ActType_No, deep, col);
  } else {
    MatMulOpt(a, b, c, bias, ActType_No, deep, row, col, col, OutType_Nhwc);
  }
}

void GruStepUnit(float *output, float *update_gate, float *reset_gate, float *hidden_buffer, const float *state_weight,
                 const float *state_bias, float *hidden_state, float *buffer[4], const GruParameter *gru_param) {
  float *packed_state = buffer[2];
  float *state_gate = buffer[3];
  bool is_vec = gru_param->batch_ == 1;

  const float *state_update_weight = state_weight;
  const float *state_reset_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_;
  const float *state_hidden_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_ * 2;
  float *state_update_gate = state_gate;
  float *state_reset_gate = state_gate + gru_param->batch_ * gru_param->hidden_size_;
  float *state_hidden_buffer = state_gate + gru_param->batch_ * gru_param->hidden_size_ * 2;
  const float *state_update_bias = state_bias;
  const float *state_reset_bias = state_bias + gru_param->hidden_size_;
  const float *state_hidden_bias = state_bias + gru_param->hidden_size_ * 2;

  // state * weight
  if (is_vec) {
    GruMatMul(state_reset_gate, hidden_state, state_reset_weight, state_reset_bias, gru_param->batch_,
              gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    GruMatMul(state_update_gate, hidden_state, state_update_weight, state_update_bias, gru_param->batch_,
              gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    PackLstmInput(hidden_state, packed_state, gru_param->batch_, gru_param->hidden_size_);
    GruMatMul(state_reset_gate, packed_state, state_reset_weight, state_reset_bias, gru_param->batch_,
              gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    GruMatMul(state_update_gate, packed_state, state_update_weight, state_update_bias, gru_param->batch_,
              gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  }
  ElementAdd(update_gate, state_update_gate, update_gate, gru_param->batch_ * gru_param->hidden_size_);
  ElementAdd(reset_gate, state_update_gate + gru_param->batch_ * gru_param->hidden_size_, reset_gate,
             gru_param->batch_ * gru_param->hidden_size_);

  // update reset_gate
  Sigmoid(reset_gate, gru_param->batch_ * gru_param->hidden_size_, reset_gate);
  // update update_gate
  Sigmoid(update_gate, gru_param->batch_ * gru_param->hidden_size_, update_gate);

  ElementMul(hidden_state, reset_gate, reset_gate, gru_param->batch_ * gru_param->hidden_size_);
  if (is_vec) {
    GruMatMul(state_hidden_buffer, reset_gate, state_hidden_weight, state_hidden_bias, gru_param->batch_,
              gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    PackLstmInput(reset_gate, packed_state, gru_param->batch_, gru_param->hidden_size_);
    GruMatMul(state_hidden_buffer, packed_state, state_hidden_weight, state_hidden_bias, gru_param->batch_,
              gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  }
  ElementAdd(hidden_buffer, state_hidden_buffer, hidden_buffer, gru_param->batch_ * gru_param->hidden_size_);

  Tanh(hidden_buffer, gru_param->batch_ * gru_param->hidden_size_, hidden_buffer);

  ElementMul(update_gate, hidden_state, hidden_state, gru_param->batch_ * gru_param->hidden_size_);

  ArithmeticParameter parameter;
  parameter.in_elements_num0_ = 1;
  parameter.in_elements_num1_ = gru_param->batch_ * gru_param->hidden_size_;
  const float one = 1.0f;
  ElementOptSub(&one, update_gate, update_gate, gru_param->batch_ * gru_param->hidden_size_, &parameter);

  ElementMulAcc(update_gate, hidden_buffer, hidden_state, gru_param->batch_ * gru_param->hidden_size_);

  memcpy(output, hidden_state, gru_param->batch_ * gru_param->hidden_size_ * sizeof(float));
}

void GruUnidirectional(float *output, const float *packed_input, const float *weight_g, const float *weight_r,
                       const float *input_bias, const float *state_bias, float *hidden_state, float *buffer[4],
                       const GruParameter *gru_param, bool is_backward) {
  float *gate = buffer[1];
  for (int i = 0; i < 3; i++) {
    const float *weight_loop = weight_g + gru_param->input_size_ * gru_param->input_col_align_ * i;
    const float *bias_loop = input_bias + gru_param->input_col_align_ * i;
    float *gate_loop = gate + gru_param->seq_len_ * gru_param->batch_ * gru_param->hidden_size_ * i;
    MatMulOpt(packed_input, weight_loop, gate_loop, bias_loop, ActType_No, gru_param->input_size_,
              gru_param->seq_len_ * gru_param->batch_, gru_param->hidden_size_, gru_param->hidden_size_, OutType_Nhwc);
  }

  float *update_gate = gate;
  float *reset_gate = gate + gru_param->seq_len_ * gru_param->batch_ * gru_param->hidden_size_;
  float *hidden_buffer = gate + gru_param->seq_len_ * gru_param->batch_ * gru_param->hidden_size_ * 2;
  for (int t = 0; t < gru_param->seq_len_; t++) {
    int real_t = is_backward ? gru_param->seq_len_ - t - 1 : t;
    float *update_gate_t = update_gate + gru_param->batch_ * gru_param->hidden_size_ * real_t;
    float *reset_gate_t = reset_gate + gru_param->batch_ * gru_param->hidden_size_ * real_t;
    float *hidden_buffer_t = hidden_buffer + gru_param->batch_ * gru_param->hidden_size_ * real_t;
    float *output_ptr = output + real_t * gru_param->output_step_;
    GruStepUnit(output_ptr, update_gate_t, reset_gate_t, hidden_buffer_t, weight_r, state_bias, hidden_state, buffer,
                gru_param);
  }
}

void Gru(float *output, const float *input, const float *weight_g, const float *weight_r, const float *input_bias,
         const float *state_bias, float *hidden_state, float *buffer[4], int check_seq_len,
         const GruParameter *gru_param) {
  // forward
  float *packed_input = buffer[0];
  PackLstmInput(input, packed_input, gru_param->seq_len_ * gru_param->batch_, gru_param->input_size_);
  GruUnidirectional(output, packed_input, weight_g, weight_r, input_bias, state_bias, hidden_state, buffer, gru_param,
                    false);

  // zero out extra fw outputs
  for (int t = check_seq_len; t < gru_param->seq_len_; t++) {
    float *output_ptr = output + t * gru_param->output_step_;
    for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
      output_ptr[i] = 0.0f;
    }
  }

  // backward
  if (gru_param->bidirectional_) {
    const float *backward_weight_g = weight_g + 3 * gru_param->input_col_align_ * gru_param->input_size_;
    const float *backward_weight_r = weight_r + 3 * gru_param->state_col_align_ * gru_param->hidden_size_;
    const float *backward_input_bias = input_bias + 3 * gru_param->input_col_align_;
    const float *backward_state_bias = state_bias + 3 * gru_param->state_col_align_;
    float *backward_output = output + gru_param->batch_ * gru_param->hidden_size_;
    float *backward_hidden_state = hidden_state + gru_param->batch_ * gru_param->hidden_size_;

    GruUnidirectional(backward_output, packed_input, backward_weight_g, backward_weight_r, backward_input_bias,
                      backward_state_bias, backward_hidden_state, buffer, gru_param, true);

    // zero out extra bw outputs
    for (int t = gru_param->seq_len_ - 1; t >= check_seq_len; t--) {
      float *output_ptr = backward_output + t * gru_param->output_step_;
      for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
        output_ptr[i] = 0.0f;
      }
    }
  }
}
