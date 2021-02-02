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

#include "nnacl/fp32/lstm_fp32.h"
#include <string.h>
#include <float.h>
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/mul_fp32.h"

void InitGate(float *gate_buffer, const float *bias, const LstmParameter *lstm_parm) {
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
      const float *input_col = input + r * inner_size;
      const float *weight_col = weight + c * inner_size;
      int index = 0;
#ifdef ENABLE_ARM
      float32x4_t out = vdupq_n_f32(0.0f);
      for (; index <= inner_size - 4; index += 4) {
        float32x4_t in_0 = vld1q_f32(input_col + index);
        float32x4_t in_1 = vld1q_f32(weight_col + index);
        out = vmlaq_f32(out, in_1, in_0);
      }
#ifdef ENABLE_ARM64
      res += vaddvq_f32(out);
#else
      float32x2_t add2 = vadd_f32(vget_low_f32(out), vget_high_f32(out));
      float32x2_t add4 = vpadd_f32(add2, add2);
      res += vget_lane_f32(add4, 0);
#endif
#endif
      for (; index < inner_size; index++) {
        res += input_col[index] * weight_col[index];
      }
      output[r * cols + c] += res;
    }
  }
}

void ElementMulAcc(const float *input0, const float *input1, float *output, int element_size) {
  int index = 0;
#ifdef ENABLE_ARM
  for (; index <= element_size - 4; index += 4) {
    float32x4_t in_0 = vld1q_f32(input0 + index);
    float32x4_t in_1 = vld1q_f32(input1 + index);
    float32x4_t out = vld1q_f32(output + index);
    out = vmlaq_f32(out, in_1, in_0);
    vst1q_f32(output + index, out);
  }
#endif
  for (; index < element_size; index++) {
    output[index] += input0[index] * input1[index];
  }
}

int ElementOptMulAcc(const float *input0, const float input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vout = vld1q_f32(output + index);
    vout = vmlaq_n_f32(vout, vin0, input1);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] += input0[index] * input1;
  }
  return NNACL_OK;
}

void UpdataState(float *cell_state, const float *forget_gate, const float *input_gate, const float *cell_gate,
                 float *state_buffer, int batch, int hidden_size, const float smooth) {
  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {  // smooth * old_cell_state
    memcpy(state_buffer, cell_state, batch * hidden_size * sizeof(float));
    ArithmeticParameter parameter;
    parameter.in_elements_num0_ = batch * hidden_size;
    parameter.in_elements_num1_ = 1;
    ElementOptMul(state_buffer, &smooth, state_buffer, batch * hidden_size, &parameter);
  }

  ElementMul(forget_gate, cell_state, cell_state, batch * hidden_size);
  ElementMulAcc(input_gate, cell_gate, cell_state, batch * hidden_size);

  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {  // (1 - smooth) * new_cell_state
    ElementOptMulAcc(cell_state, 1 - smooth, state_buffer, batch * hidden_size);
  }
}

void UpdataOutput(const float *cell_state, const float *output_gate, float *hidden_state, float *state_buffer_in,
                  int batch, int hidden_size, const float smooth) {
  float *state_buffer = state_buffer_in + batch * hidden_size;
  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {
    memcpy(state_buffer, hidden_state, batch * hidden_size * sizeof(float));
    ArithmeticParameter parameter;
    parameter.in_elements_num0_ = batch * hidden_size;
    parameter.in_elements_num1_ = 1;
    ElementOptMul(state_buffer, &smooth, state_buffer, batch * hidden_size, &parameter);
  }

  Tanh(cell_state, batch * hidden_size, hidden_state);
  ElementMul(hidden_state, output_gate, hidden_state, batch * hidden_size);

  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {
    ElementOptMulAcc(hidden_state, 1 - smooth, state_buffer, batch * hidden_size);
  }
}

void LstmStepUnit(float *output, const float *input, const float *input_input_weight, const float *input_forget_weight,
                  const float *input_cell_weight, const float *input_output_weight, const float *state_input_weight,
                  const float *state_forget_weight, const float *state_cell_weight, const float *state_output_weight,
                  const float *bias, float *hidden_state, float *cell_state, float *gate_buffer, float *state_buffer,
                  const LstmParameter *lstm_parm) {
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
  UpdataState(cell_state, forget_gate, input_gate, cell_gate, state_buffer, lstm_parm->batch_, lstm_parm->hidden_size_,
              lstm_parm->smooth_);

  // update output_gate
  Sigmoid(output_gate, lstm_parm->batch_ * lstm_parm->hidden_size_, output_gate);
  // update output
  UpdataOutput(cell_state, output_gate, hidden_state, state_buffer, lstm_parm->batch_, lstm_parm->hidden_size_,
               lstm_parm->smooth_);
  memcpy(output, hidden_state, lstm_parm->batch_ * lstm_parm->hidden_size_ * sizeof(float));

  if (!(lstm_parm->smooth_ >= -FLT_EPSILON && lstm_parm->smooth_ <= FLT_EPSILON)) {
    memcpy(cell_state, state_buffer, lstm_parm->batch_ * lstm_parm->hidden_size_ * sizeof(float));
    memcpy(hidden_state, state_buffer + lstm_parm->batch_ * lstm_parm->hidden_size_,
           lstm_parm->batch_ * lstm_parm->hidden_size_ * sizeof(float));
  }
}

void Lstm(float *output, const float *input, const float *weight_i, const float *weight_h, const float *bias,
          float *hidden_state, float *cell_state, float *gate_buffer, float *state_buffer,
          const LstmParameter *lstm_parm) {
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
                 cell_state, gate_buffer, state_buffer, lstm_parm);
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
                   backward_bias, backward_hidden_state, backward_cell_state, gate_buffer, state_buffer, lstm_parm);
    }
  }
}
