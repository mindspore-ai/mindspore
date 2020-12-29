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

void InitGruGate(float *gate_buffer, const float *bias, const GruParameter *gru_parm) {
  int gate_offest = 0;
  for (int l = 0; l < 3; l++) {
    int batch_offest = gate_offest;
    int bias_offest = l * gru_parm->hidden_size_;
    for (int b = 0; b < gru_parm->batch_; b++) {
      memcpy(gate_buffer + batch_offest, bias + bias_offest, gru_parm->hidden_size_ * sizeof(float));
      batch_offest += gru_parm->hidden_size_;
    }
    gate_offest += gru_parm->batch_ * gru_parm->hidden_size_;
  }
}

void GruStepUnit(float *output, const float *input, const float *input_reset_weight, const float *input_update_weight,
                 const float *input_hidden_weight, const float *state_reset_weight, const float *state_update_weight,
                 const float *state_hidden_weight, const float *bias, float *hidden_state, float *gate_buffer,
                 const GruParameter *gru_parm) {
  InitGruGate(gate_buffer, bias, gru_parm);

  float *update_gate = gate_buffer;
  float *reset_gate = gate_buffer + gru_parm->batch_ * gru_parm->hidden_size_;
  float *hidden_buffer = gate_buffer + gru_parm->batch_ * gru_parm->hidden_size_ * 2;

  // input * weight
  MatMulAcc(reset_gate, input, input_reset_weight, gru_parm->batch_, gru_parm->hidden_size_, gru_parm->input_size_);
  MatMulAcc(update_gate, input, input_update_weight, gru_parm->batch_, gru_parm->hidden_size_, gru_parm->input_size_);
  MatMulAcc(hidden_buffer, input, input_hidden_weight, gru_parm->batch_, gru_parm->hidden_size_, gru_parm->input_size_);

  // state * weight
  MatMulAcc(reset_gate, hidden_state, state_reset_weight, gru_parm->batch_, gru_parm->hidden_size_,
            gru_parm->hidden_size_);
  MatMulAcc(update_gate, hidden_state, state_update_weight, gru_parm->batch_, gru_parm->hidden_size_,
            gru_parm->hidden_size_);

  // update reset_gate
  Sigmoid(reset_gate, gru_parm->batch_ * gru_parm->hidden_size_, reset_gate);

  // update update_gate
  Sigmoid(update_gate, gru_parm->batch_ * gru_parm->hidden_size_, update_gate);

  ElementMul(hidden_state, reset_gate, reset_gate, gru_parm->batch_ * gru_parm->hidden_size_);
  MatMulAcc(hidden_buffer, reset_gate, state_hidden_weight, gru_parm->batch_, gru_parm->hidden_size_,
            gru_parm->hidden_size_);

  Tanh(hidden_buffer, gru_parm->batch_ * gru_parm->hidden_size_, hidden_buffer);

  ElementMul(update_gate, hidden_state, hidden_state, gru_parm->batch_ * gru_parm->hidden_size_);

  ArithmeticParameter parameter;
  parameter.in_elements_num0_ = 1;
  parameter.in_elements_num1_ = gru_parm->batch_ * gru_parm->hidden_size_;
  const float one = 1.0f;
  ElementOptSub(&one, update_gate, update_gate, gru_parm->batch_ * gru_parm->hidden_size_, &parameter);

  ElementMulAcc(update_gate, hidden_buffer, hidden_state, gru_parm->batch_ * gru_parm->hidden_size_);

  memcpy(output, hidden_state, gru_parm->batch_ * gru_parm->hidden_size_ * sizeof(float));
}

void Gru(float *output, const float *input, const float *weight_g, const float *weight_r, const float *bias,
         float *hidden_state, float *gate_buffer, int check_seq_len, const GruParameter *gru_parm) {
  // forward
  const float *input_update_weight = weight_g;
  const float *input_reset_weight = weight_g + gru_parm->input_size_ * gru_parm->hidden_size_;
  const float *input_hidden_weight = weight_g + gru_parm->input_size_ * gru_parm->hidden_size_ * 2;

  const float *state_update_weight = weight_r;
  const float *state_reset_weight = weight_r + gru_parm->hidden_size_ * gru_parm->hidden_size_;
  const float *state_hidden_weight = weight_r + gru_parm->hidden_size_ * gru_parm->hidden_size_ * 2;

  for (int t = 0; t < check_seq_len; t++) {
    const float *input_ptr = input + t * gru_parm->input_step_;
    float *output_ptr = output + t * gru_parm->output_step_;
    GruStepUnit(output_ptr, input_ptr, input_reset_weight, input_update_weight, input_hidden_weight, state_reset_weight,
                state_update_weight, state_hidden_weight, bias, hidden_state, gate_buffer, gru_parm);
  }
  // zero out extra fw outputs
  for (int t = check_seq_len; t < gru_parm->seq_len_; t++) {
    float *output_ptr = output + t * gru_parm->output_step_;
    for (int i = 0; i < gru_parm->batch_ * gru_parm->hidden_size_; i++) {
      output_ptr[i] = 0.0f;
    }
  }

  // backward
  if (gru_parm->bidirectional_) {
    input_update_weight = weight_g + gru_parm->input_size_ * gru_parm->hidden_size_ * 3;
    input_reset_weight = weight_g + gru_parm->input_size_ * gru_parm->hidden_size_ * 4;
    input_hidden_weight = weight_g + gru_parm->input_size_ * gru_parm->hidden_size_ * 5;

    state_update_weight = weight_r + gru_parm->hidden_size_ * gru_parm->hidden_size_ * 3;
    state_reset_weight = weight_r + gru_parm->hidden_size_ * gru_parm->hidden_size_ * 4;
    state_hidden_weight = weight_r + gru_parm->hidden_size_ * gru_parm->hidden_size_ * 5;

    float *backward_output = output + gru_parm->batch_ * gru_parm->hidden_size_;
    const float *backward_bias = bias + 3 * gru_parm->hidden_size_;
    float *backward_hidden_state = hidden_state + gru_parm->batch_ * gru_parm->hidden_size_;
    for (int t = check_seq_len - 1; t >= 0; t--) {
      const float *input_ptr = input + t * gru_parm->input_step_;
      float *output_ptr = backward_output + t * gru_parm->output_step_;
      GruStepUnit(output_ptr, input_ptr, input_reset_weight, input_update_weight, input_hidden_weight,
                  state_reset_weight, state_update_weight, state_hidden_weight, backward_bias, backward_hidden_state,
                  gate_buffer, gru_parm);
    }
    // zero out extra bw outputs
    for (int t = gru_parm->seq_len_ - 1; t >= check_seq_len; t--) {
      float *output_ptr = backward_output + t * gru_parm->output_step_;
      for (int i = 0; i < gru_parm->batch_ * gru_parm->hidden_size_; i++) {
        output_ptr[i] = 0.0f;
      }
    }
  }
}
