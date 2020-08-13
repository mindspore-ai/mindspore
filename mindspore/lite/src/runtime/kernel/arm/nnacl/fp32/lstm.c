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

#include "nnacl/fp32/lstm.h"
#include <string.h>
#include "nnacl/fp32/activation.h"
#include "nnacl/fp32/arithmetic.h"

void InitGate(float *gate_buffer, const float *bias, LstmParameter *lstm_parm) {
  int gate_offest = 0;
  for (int l = 0; l < 4; l++) {
    int batch_offest = gate_offest;
    int bias_offest = l * lstm_parm->hidden_size_;
    for (int b = 0; b < lstm_parm->batch_; b++) {
      memcpy(gate_buffer + batch_offest, bias + bias_offest, lstm_parm->hidden_size_ * sizeof(float));
      batch_offest += lstm_parm->hidden_size_;
    }
    gate_offest += lstm_parm->batch_ * lstm_parm->hidden_size_;
  }
}

// input: [row, inner_size]; weight: [col, inner_size]; output: [row, col]
void MatMulAcc(float *output, const float *input, const float *weight, int rows, int cols, int inner_size) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float res = 0;
      for (int i = 0; i < inner_size; i++) {
        res += input[r * inner_size + i] * weight[c * inner_size + i];
      }
      output[r * cols + c] += res;
    }
  }
}

void ElementMulAcc(float *input0, float *input1, float *output, int element_size) {
  for (int index = 0; index < element_size; index++) {
    output[index] += input0[index] * input1[index];
  }
}

void UpdataState(float *cell_state, float *forget_gate, float *input_gate, float *cell_gate, int batch,
                 int hidden_size) {
  ElementMul(forget_gate, cell_state, cell_state, batch * hidden_size);
  ElementMulAcc(input_gate, cell_gate, cell_state, batch * hidden_size);
}

void UpdataOutput(float *cell_state, float *output_gate, float *hidden_state, int batch, int hidden_size) {
  Tanh(cell_state, batch * hidden_size, hidden_state);
  ElementMul(hidden_state, output_gate, hidden_state, batch * hidden_size);
}

void LstmStepUnit(float *output, const float *input, const float *input_input_weight, const float *input_forget_weight,
                  const float *input_cell_weight, const float *input_output_weight, const float *state_input_weight,
                  const float *state_forget_weight, const float *state_cell_weight, const float *state_output_weight,
                  const float *bias, float *hidden_state, float *cell_state, float *gate_buffer,
                  LstmParameter *lstm_parm) {
  InitGate(gate_buffer, bias, lstm_parm);

  float *input_gate = gate_buffer;
  float *forget_gate = gate_buffer + lstm_parm->batch_ * lstm_parm->hidden_size_ * 2;
  float *cell_gate = gate_buffer + lstm_parm->batch_ * lstm_parm->hidden_size_ * 3;
  float *output_gate = gate_buffer + lstm_parm->batch_ * lstm_parm->hidden_size_ * 1;

  // input * weight
  MatMulAcc(input_gate, input, input_input_weight, lstm_parm->batch_, lstm_parm->hidden_size_, lstm_parm->input_size_);
  MatMulAcc(forget_gate, input, input_forget_weight, lstm_parm->batch_, lstm_parm->hidden_size_,
            lstm_parm->input_size_);
  MatMulAcc(cell_gate, input, input_cell_weight, lstm_parm->batch_, lstm_parm->hidden_size_, lstm_parm->input_size_);
  MatMulAcc(output_gate, input, input_output_weight, lstm_parm->batch_, lstm_parm->hidden_size_,
            lstm_parm->input_size_);

  // state * weight
  MatMulAcc(input_gate, hidden_state, state_input_weight, lstm_parm->batch_, lstm_parm->hidden_size_,
            lstm_parm->hidden_size_);
  MatMulAcc(forget_gate, hidden_state, state_forget_weight, lstm_parm->batch_, lstm_parm->hidden_size_,
            lstm_parm->hidden_size_);
  MatMulAcc(cell_gate, hidden_state, state_cell_weight, lstm_parm->batch_, lstm_parm->hidden_size_,
            lstm_parm->hidden_size_);
  MatMulAcc(output_gate, hidden_state, state_output_weight, lstm_parm->batch_, lstm_parm->hidden_size_,
            lstm_parm->hidden_size_);

  // update input_gate
  Sigmoid(input_gate, lstm_parm->batch_ * lstm_parm->hidden_size_, input_gate);

  // update forget_gate
  Sigmoid(forget_gate, lstm_parm->batch_ * lstm_parm->hidden_size_, forget_gate);

  // update cell_gate
  Tanh(cell_gate, lstm_parm->batch_ * lstm_parm->hidden_size_, cell_gate);
  // update cell state
  UpdataState(cell_state, forget_gate, input_gate, cell_gate, lstm_parm->batch_, lstm_parm->hidden_size_);

  // update output_gate
  Sigmoid(output_gate, lstm_parm->batch_ * lstm_parm->hidden_size_, output_gate);
  // update output
  UpdataOutput(cell_state, output_gate, hidden_state, lstm_parm->batch_, lstm_parm->hidden_size_);
  memcpy(output, hidden_state, lstm_parm->batch_ * lstm_parm->hidden_size_ * sizeof(float));
}

void Lstm(float *output, const float *input, const float *weight_i, const float *weight_h, const float *bias,
          float *hidden_state, float *cell_state, float *gate_buffer, LstmParameter *lstm_parm) {
  // forward
  const float *input_input_weight = weight_i;
  const float *input_forget_weight = weight_i + lstm_parm->input_size_ * lstm_parm->hidden_size_ * 2;
  const float *input_cell_weight = weight_i + lstm_parm->input_size_ * lstm_parm->hidden_size_ * 3;
  const float *input_output_weight = weight_i + lstm_parm->input_size_ * lstm_parm->hidden_size_ * 1;

  const float *state_input_weight = weight_h;
  const float *state_forget_weight = weight_h + lstm_parm->hidden_size_ * lstm_parm->hidden_size_ * 2;
  const float *state_cell_weight = weight_h + lstm_parm->hidden_size_ * lstm_parm->hidden_size_ * 3;
  const float *state_output_weight = weight_h + lstm_parm->hidden_size_ * lstm_parm->hidden_size_ * 1;

  for (int t = 0; t < lstm_parm->seq_len_; t++) {
    const float *input_ptr = input + t * lstm_parm->input_step_;
    float *output_ptr = output + t * lstm_parm->output_step_;
    LstmStepUnit(output_ptr, input_ptr, input_input_weight, input_forget_weight, input_cell_weight, input_output_weight,
                 state_input_weight, state_forget_weight, state_cell_weight, state_output_weight, bias, hidden_state,
                 cell_state, gate_buffer, lstm_parm);
  }

  // backward
  if (lstm_parm->bidirectional_) {
    input_input_weight = weight_i + lstm_parm->input_size_ * lstm_parm->hidden_size_ * 4;
    input_forget_weight = weight_i + lstm_parm->input_size_ * lstm_parm->hidden_size_ * 6;
    input_cell_weight = weight_i + lstm_parm->input_size_ * lstm_parm->hidden_size_ * 7;
    input_output_weight = weight_i + lstm_parm->input_size_ * lstm_parm->hidden_size_ * 5;

    state_input_weight = weight_h + lstm_parm->hidden_size_ * lstm_parm->hidden_size_ * 4;
    state_forget_weight = weight_h + lstm_parm->hidden_size_ * lstm_parm->hidden_size_ * 6;
    state_cell_weight = weight_h + lstm_parm->hidden_size_ * lstm_parm->hidden_size_ * 7;
    state_output_weight = weight_h + lstm_parm->hidden_size_ * lstm_parm->hidden_size_ * 5;

    float *backward_output = output + lstm_parm->batch_ * lstm_parm->hidden_size_;
    const float *backward_bias = bias + 4 * lstm_parm->hidden_size_;
    float *backward_cell_state = cell_state + lstm_parm->batch_ * lstm_parm->hidden_size_;
    float *backward_hidden_state = hidden_state + lstm_parm->batch_ * lstm_parm->hidden_size_;
    for (int t = lstm_parm->seq_len_ - 1; t >= 0; t--) {
      const float *input_ptr = input + t * lstm_parm->input_step_;
      float *output_ptr = backward_output + t * lstm_parm->output_step_;
      LstmStepUnit(output_ptr, input_ptr, input_input_weight, input_forget_weight, input_cell_weight,
                   input_output_weight, state_input_weight, state_forget_weight, state_cell_weight, state_output_weight,
                   backward_bias, backward_hidden_state, backward_cell_state, gate_buffer, lstm_parm);
    }
  }
}
