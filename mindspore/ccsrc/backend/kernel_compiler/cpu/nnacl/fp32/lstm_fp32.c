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
#include "nnacl/fp32/matmul_fp32.h"

void PackLstmWeight(float *dst, const float *src, int batch, int deep, int col, int col_align) {
  for (int i = 0; i < batch; i++) {
    const float *src_batch = src + i * col * deep;
    float *dst_batch = dst + i * col_align * deep;
#ifdef ENABLE_AVX
    RowMajor2Col16Major(src_batch, dst_batch, col, deep);
#elif defined(ENABLE_ARM32)
    RowMajor2Col4Major(src_batch, dst_batch, col, deep);
#else
    RowMajor2Col8Major(src_batch, dst_batch, col, deep);
#endif
  }
}

void PackLstmBias(float *dst, const float *src, int batch, int col, int col_align, bool is_bidirectional) {
  int unidirectional_batch = is_bidirectional ? batch / 2 : batch;
  for (int i = 0; i < unidirectional_batch; i++) {
    const float *src_batch = src + i * col;
    float *dst_batch = dst + i * col_align;
    memcpy(dst_batch, src_batch, col * sizeof(float));
  }
  if (is_bidirectional) {
    const float *backward_src = src + batch * col;
    float *backward_dst = dst + unidirectional_batch * col_align;
    for (int i = 0; i < unidirectional_batch; i++) {
      const float *backward_src_batch = backward_src + i * col;
      float *backward_dst_batch = backward_dst + i * col_align;
      memcpy(backward_dst_batch, backward_src_batch, col * sizeof(float));
    }
  }
}

void PackLstmInput(const float *src, float *dst, int row, int deep) {
#ifdef ENABLE_AVX
  RowMajor2Col6Major(src, dst, row, deep);
#elif defined(ENABLE_SSE)
  RowMajor2Col4Major(src, dst, row, deep);
#else
  RowMajor2Col12Major(src, dst, row, deep);
#endif
}

void LstmMatMul(float *c, const float *a, const float *b, const float *bias, int row, int deep, int col, bool is_vec) {
  if (is_vec) {
    MatVecMulFp32(a, b, c, bias, ActType_No, deep, col);
  } else {
    MatMulOpt(a, b, c, bias, ActType_No, deep, row, col, col, OutType_Nhwc);
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
                 float *state_buffer, int batch, int hidden_size, const float zoneout) {
  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {  // zoneout * old_cell_state
    memcpy(state_buffer, cell_state, batch * hidden_size * sizeof(float));
    ArithmeticParameter parameter;
    parameter.in_elements_num0_ = batch * hidden_size;
    parameter.in_elements_num1_ = 1;
    ElementOptMul(state_buffer, &zoneout, state_buffer, batch * hidden_size, &parameter);
  }

  ElementMul(forget_gate, cell_state, cell_state, batch * hidden_size);
  ElementMulAcc(input_gate, cell_gate, cell_state, batch * hidden_size);

  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {  // (1 - zoneout) * new_cell_state
    ElementOptMulAcc(cell_state, 1 - zoneout, state_buffer, batch * hidden_size);
  }
}

void UpdataOutput(const float *cell_state, const float *output_gate, float *hidden_state, float *state_buffer,
                  int batch, int hidden_size, const float zoneout) {
  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {
    memcpy(state_buffer, hidden_state, batch * hidden_size * sizeof(float));
    ArithmeticParameter parameter;
    parameter.in_elements_num0_ = batch * hidden_size;
    parameter.in_elements_num1_ = 1;
    ElementOptMul(state_buffer, &zoneout, state_buffer, batch * hidden_size, &parameter);
  }

  Tanh(cell_state, batch * hidden_size, hidden_state);
  ElementMul(hidden_state, output_gate, hidden_state, batch * hidden_size);

  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {
    ElementOptMulAcc(hidden_state, 1 - zoneout, state_buffer, batch * hidden_size);
  }
}

void UpdateLstmGate(float *gate_buffer, const float *input, const float *weight, const float *bias, int row, int deep,
                    int col, int col_align, bool is_vec) {
  for (int i = 0; i < 4; i++) {
    const float *weight_i = weight + deep * col * i;
    const float *bias_i = bias + col_align * i;
    float *gate = gate_buffer + row * col * i;
    LstmMatMul(gate, input, weight_i, bias_i, row, deep, col, is_vec);
  }
}

void LstmStepUnit(float *output, float *input_gate, float *forget_gate, float *cell_gate, float *output_gate,
                  const float *state_weight, const float *state_bias, float *hidden_state, float *cell_state,
                  float *buffer[6], const LstmParameter *lstm_param) {
  float *packed_state = buffer[2];
  float *state_gate = buffer[3];
  float *cell_buffer = buffer[4];
  float *hidden_buffer = buffer[5];
  bool is_vec = lstm_param->batch_ == 1;
  // state * weight
  if (is_vec) {
    UpdateLstmGate(state_gate, hidden_state, state_weight, state_bias, lstm_param->batch_, lstm_param->hidden_size_,
                   lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec);
  } else {
    // pack state for matmul
    PackLstmInput(hidden_state, packed_state, lstm_param->batch_, lstm_param->hidden_size_);
    UpdateLstmGate(state_gate, packed_state, state_weight, state_bias, lstm_param->batch_, lstm_param->hidden_size_,
                   lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec);
  }
  ElementAdd(input_gate, state_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_);
  ElementAdd(forget_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 2, forget_gate,
             lstm_param->batch_ * lstm_param->hidden_size_);
  ElementAdd(cell_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 3, cell_gate,
             lstm_param->batch_ * lstm_param->hidden_size_);
  ElementAdd(output_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_, output_gate,
             lstm_param->batch_ * lstm_param->hidden_size_);

  // update input_gate
  Sigmoid(input_gate, lstm_param->batch_ * lstm_param->hidden_size_, input_gate);

  // update forget_gate
  Sigmoid(forget_gate, lstm_param->batch_ * lstm_param->hidden_size_, forget_gate);

  // update cell_gate
  Tanh(cell_gate, lstm_param->batch_ * lstm_param->hidden_size_, cell_gate);
  // update cell state
  UpdataState(cell_state, forget_gate, input_gate, cell_gate, cell_buffer, lstm_param->batch_, lstm_param->hidden_size_,
              lstm_param->zoneout_cell_);

  // update output_gate
  Sigmoid(output_gate, lstm_param->batch_ * lstm_param->hidden_size_, output_gate);
  // update output
  UpdataOutput(cell_state, output_gate, hidden_state, hidden_buffer, lstm_param->batch_, lstm_param->hidden_size_,
               lstm_param->zoneout_hidden_);
  memcpy(output, hidden_state, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));

  if (!(lstm_param->zoneout_cell_ >= -FLT_EPSILON && lstm_param->zoneout_cell_ <= FLT_EPSILON)) {
    memcpy(cell_state, cell_buffer, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));
  }

  if (!(lstm_param->zoneout_hidden_ >= -FLT_EPSILON && lstm_param->zoneout_hidden_ <= FLT_EPSILON)) {
    memcpy(hidden_state, hidden_buffer, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));
  }
}

void LstmUnidirectional(float *output, const float *packed_input, const float *weight_i, const float *weight_h,
                        const float *input_bias, const float *state_bias, float *hidden_state, float *cell_state,
                        float *buffer[6], const LstmParameter *lstm_param, bool is_backward) {
  float *gate = buffer[1];
  for (int i = 0; i < 4; i++) {
    const float *weight_loop = weight_i + lstm_param->input_size_ * lstm_param->input_col_align_ * i;
    const float *bias_loop = input_bias + lstm_param->input_col_align_ * i;
    float *gate_loop = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * i;
    MatMulOpt(packed_input, weight_loop, gate_loop, bias_loop, ActType_No, lstm_param->input_size_,
              lstm_param->seq_len_ * lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_,
              OutType_Nhwc);
  }

  float *input_gate = gate;
  float *forget_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 2;
  float *cell_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 3;
  float *output_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_;
  for (int t = 0; t < lstm_param->seq_len_; t++) {
    int real_t = is_backward ? lstm_param->seq_len_ - t - 1 : t;
    float *input_gate_t = input_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float *forget_gate_t = forget_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float *cell_gate_t = cell_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float *output_gate_t = output_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float *output_ptr = output + real_t * lstm_param->output_step_;
    LstmStepUnit(output_ptr, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias,
                 hidden_state, cell_state, buffer, lstm_param);
  }
}

void Lstm(float *output, const float *input, const float *weight_i, const float *weight_h, const float *input_bias,
          const float *state_bias, float *hidden_state, float *cell_state, float *buffer[6],
          const LstmParameter *lstm_param) {
  // forward
  float *packed_input = buffer[0];
  PackLstmInput(input, packed_input, lstm_param->seq_len_ * lstm_param->batch_, lstm_param->input_size_);
  LstmUnidirectional(output, packed_input, weight_i, weight_h, input_bias, state_bias, hidden_state, cell_state, buffer,
                     lstm_param, false);

  // backward
  if (lstm_param->bidirectional_) {
    const float *backward_weight_i = weight_i + 4 * lstm_param->input_col_align_ * lstm_param->input_size_;
    const float *backward_weight_h = weight_h + 4 * lstm_param->state_col_align_ * lstm_param->hidden_size_;
    const float *backward_input_bias = input_bias + 4 * lstm_param->input_col_align_;
    const float *backward_state_bias = state_bias + 4 * lstm_param->state_col_align_;
    float *backward_output = output + lstm_param->batch_ * lstm_param->hidden_size_;
    float *backward_cell_state = cell_state + lstm_param->batch_ * lstm_param->hidden_size_;
    float *backward_hidden_state = hidden_state + lstm_param->batch_ * lstm_param->hidden_size_;

    LstmUnidirectional(backward_output, packed_input, backward_weight_i, backward_weight_h, backward_input_bias,
                       backward_state_bias, backward_hidden_state, backward_cell_state, buffer, lstm_param, true);
  }
}
