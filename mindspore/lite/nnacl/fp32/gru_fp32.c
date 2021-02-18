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

void UpdateGruInputGate(float *gate_buffer, const float *input, const float *weight, const float *bias, int row,
                        int deep, int col, int col_align, bool is_vec) {
  for (int i = 0; i < 3; i++) {
    const float *weight_i = weight + deep * col * i;
    const float *bias_i = bias + col_align * i;
    float *gate = gate_buffer + row * col * i;
    LstmMatMul(gate, input, weight_i, bias_i, row, deep, col, is_vec);
  }
}

void GruStepUnit(float *output, const float *input, const float *input_weight, const float *state_weight,
                 const float *bias, float *hidden_state, float *gate_buffer, float *matmul_buffer[2],
                 const GruParameter *gru_param) {
  bool is_vec = gru_param->batch_ == 1;
  // input * weight
  if (is_vec) {
    UpdateGruInputGate(gate_buffer, input, input_weight, bias, gru_param->batch_, gru_param->input_size_,
                       gru_param->hidden_size_, gru_param->col_align_, is_vec);
  } else {
    // pack input for matmul
    PackLstmInput(input, matmul_buffer[0], gru_param->batch_, gru_param->input_size_);
    UpdateGruInputGate(gate_buffer, matmul_buffer[0], input_weight, bias, gru_param->batch_, gru_param->input_size_,
                       gru_param->hidden_size_, gru_param->col_align_, is_vec);
  }

  const float *state_update_weight = state_weight;
  const float *state_reset_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_;
  const float *state_hidden_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_ * 2;
  float *state_update_gate = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 3;
  float *state_reset_gate = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 4;
  float *state_hidden_buffer = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 5;
  const float *state_update_bias = bias + gru_param->hidden_size_ * 3;
  const float *state_reset_bias = bias + gru_param->hidden_size_ * 4;
  const float *state_hidden_bias = bias + gru_param->hidden_size_ * 5;

  // state * weight
  if (is_vec) {
    LstmMatMul(state_reset_gate, hidden_state, state_reset_weight, state_reset_bias, gru_param->batch_,
               gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    LstmMatMul(state_update_gate, hidden_state, state_update_weight, state_update_bias, gru_param->batch_,
               gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    PackLstmInput(hidden_state, matmul_buffer[1], gru_param->batch_, gru_param->hidden_size_);
    LstmMatMul(state_reset_gate, matmul_buffer[1], state_reset_weight, state_reset_bias, gru_param->batch_,
               gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    LstmMatMul(state_update_gate, matmul_buffer[1], state_update_weight, state_update_bias, gru_param->batch_,
               gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  }
  ElementAdd(gate_buffer, state_update_gate, gate_buffer, gru_param->batch_ * gru_param->hidden_size_ * 2);
  float *update_gate = gate_buffer;
  float *reset_gate = gate_buffer + gru_param->batch_ * gru_param->hidden_size_;
  float *hidden_buffer = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 2;

  // update reset_gate
  Sigmoid(reset_gate, gru_param->batch_ * gru_param->hidden_size_, reset_gate);
  // update update_gate
  Sigmoid(update_gate, gru_param->batch_ * gru_param->hidden_size_, update_gate);

  ElementMul(hidden_state, reset_gate, reset_gate, gru_param->batch_ * gru_param->hidden_size_);
  if (is_vec) {
    LstmMatMul(state_hidden_buffer, reset_gate, state_hidden_weight, state_hidden_bias, gru_param->batch_,
               gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    PackLstmInput(reset_gate, matmul_buffer[1], gru_param->batch_, gru_param->hidden_size_);
    LstmMatMul(state_hidden_buffer, matmul_buffer[1], state_hidden_weight, state_hidden_bias, gru_param->batch_,
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

void Gru(float *output, const float *input, const float *weight_g, const float *weight_r, const float *bias,
         float *hidden_state, float *gate_buffer, float *matmul_buffer[2], int check_seq_len,
         const GruParameter *gru_param) {
  // forward
  for (int t = 0; t < check_seq_len; t++) {
    const float *input_ptr = input + t * gru_param->input_step_;
    float *output_ptr = output + t * gru_param->output_step_;
    GruStepUnit(output_ptr, input_ptr, weight_g, weight_r, bias, hidden_state, gate_buffer, matmul_buffer, gru_param);
  }
  // zero out extra fw outputs
  for (int t = check_seq_len; t < gru_param->seq_len_; t++) {
    float *output_ptr = output + t * gru_param->output_step_;
    for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
      output_ptr[i] = 0.0f;
    }
  }

  // backward
  if (gru_param->bidirectional_) {
    const float *backward_weight_g = weight_g + 3 * gru_param->col_align_ * gru_param->input_size_;
    const float *backward_weight_r = weight_r + 3 * gru_param->col_align_ * gru_param->hidden_size_;
    const float *backward_bias = bias + 6 * gru_param->hidden_size_;
    float *backward_output = output + gru_param->batch_ * gru_param->hidden_size_;
    float *backward_hidden_state = hidden_state + gru_param->batch_ * gru_param->hidden_size_;
    for (int t = check_seq_len - 1; t >= 0; t--) {
      const float *input_ptr = input + t * gru_param->input_step_;
      float *output_ptr = backward_output + t * gru_param->output_step_;
      GruStepUnit(output_ptr, input_ptr, backward_weight_g, backward_weight_r, backward_bias, backward_hidden_state,
                  gate_buffer, matmul_buffer, gru_param);
    }
    // zero out extra bw outputs
    for (int t = gru_param->seq_len_ - 1; t >= check_seq_len; t--) {
      float *output_ptr = backward_output + t * gru_param->output_step_;
      for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
        output_ptr[i] = 0.0f;
      }
    }
  }
}
