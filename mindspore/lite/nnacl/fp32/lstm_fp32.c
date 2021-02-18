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

void PackLstmInput(const float *src, float *dst, int row, int deep) {
#ifdef ENABLE_AVX
  RowMajor2Col6Major(src, dst, row, deep);
#elif defined(ENABLE_SSE)
  RowMajor2Col4Major(src, dst, row, deep);
#else
  RowMajor2Col12Major(src, dst, row, deep);
#endif
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

void LstmMatMul(float *c, const float *a, const float *b, const float *bias, int row, int deep, int col, bool is_vec) {
  if (is_vec) {
    memcpy(c, bias, col * sizeof(float));
    MatMulAcc(c, a, b, row, col, deep);
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

void UpdateLstmGate(float *gate_buffer, const float *input, const float *weight, const float *bias, int row, int deep,
                    int col, int col_align, bool is_vec) {
  for (int i = 0; i < 4; i++) {
    const float *weight_i = weight + deep * col * i;
    const float *bias_i = bias + col_align * i;
    float *gate = gate_buffer + row * col * i;
    LstmMatMul(gate, input, weight_i, bias_i, row, deep, col, is_vec);
  }
}

void LstmStepUnit(float *output, const float *input, const float *input_weight, const float *state_weight,
                  const float *bias, float *hidden_state, float *cell_state, float *gate_buffer, float *state_buffer,
                  float *matmul_buffer[2], const LstmParameter *lstm_param) {
  bool is_vec = lstm_param->batch_ == 1;
  // input * weight
  if (is_vec) {
    UpdateLstmGate(gate_buffer, input, input_weight, bias, lstm_param->batch_, lstm_param->input_size_,
                   lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  } else {
    // pack input for matmul
    PackLstmInput(input, matmul_buffer[0], lstm_param->batch_, lstm_param->input_size_);
    UpdateLstmGate(gate_buffer, matmul_buffer[0], input_weight, bias, lstm_param->batch_, lstm_param->input_size_,
                   lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  }

  // state * weight
  float *state_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_ * 4;
  const float *state_bias = bias + lstm_param->col_align_ * 4;
  if (is_vec) {
    UpdateLstmGate(state_gate, hidden_state, state_weight, state_bias, lstm_param->batch_, lstm_param->hidden_size_,
                   lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  } else {
    // pack state for matmul
    PackLstmInput(hidden_state, matmul_buffer[1], lstm_param->batch_, lstm_param->hidden_size_);
    UpdateLstmGate(state_gate, matmul_buffer[1], state_weight, state_bias, lstm_param->batch_, lstm_param->hidden_size_,
                   lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  }
  ElementAdd(gate_buffer, state_gate, gate_buffer, 4 * lstm_param->batch_ * lstm_param->hidden_size_);

  float *input_gate = gate_buffer;
  float *forget_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_ * 2;
  float *cell_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_ * 3;
  float *output_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_;
  // update input_gate
  Sigmoid(input_gate, lstm_param->batch_ * lstm_param->hidden_size_, input_gate);

  // update forget_gate
  Sigmoid(forget_gate, lstm_param->batch_ * lstm_param->hidden_size_, forget_gate);

  // update cell_gate
  Tanh(cell_gate, lstm_param->batch_ * lstm_param->hidden_size_, cell_gate);
  // update cell state
  UpdataState(cell_state, forget_gate, input_gate, cell_gate, state_buffer, lstm_param->batch_,
              lstm_param->hidden_size_, lstm_param->smooth_);

  // update output_gate
  Sigmoid(output_gate, lstm_param->batch_ * lstm_param->hidden_size_, output_gate);
  // update output
  UpdataOutput(cell_state, output_gate, hidden_state, state_buffer, lstm_param->batch_, lstm_param->hidden_size_,
               lstm_param->smooth_);
  memcpy(output, hidden_state, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));

  if (!(lstm_param->smooth_ >= -FLT_EPSILON && lstm_param->smooth_ <= FLT_EPSILON)) {
    memcpy(cell_state, state_buffer, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));
    memcpy(hidden_state, state_buffer + lstm_param->batch_ * lstm_param->hidden_size_,
           lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));
  }
}

void Lstm(float *output, const float *input, const float *weight_i, const float *weight_h, const float *bias,
          float *hidden_state, float *cell_state, float *gate_buffer, float *state_buffer, float *matmul_buffer[2],
          const LstmParameter *lstm_param) {
  // forward
  for (int t = 0; t < lstm_param->seq_len_; t++) {
    const float *input_ptr = input + t * lstm_param->input_step_;
    float *output_ptr = output + t * lstm_param->output_step_;
    LstmStepUnit(output_ptr, input_ptr, weight_i, weight_h, bias, hidden_state, cell_state, gate_buffer, state_buffer,
                 matmul_buffer, lstm_param);
  }

  // backward
  if (lstm_param->bidirectional_) {
    const float *backward_weight_i = weight_i + 4 * lstm_param->col_align_ * lstm_param->input_size_;
    const float *backward_weight_h = weight_h + 4 * lstm_param->col_align_ * lstm_param->hidden_size_;
    const float *backward_bias = bias + 8 * lstm_param->col_align_;
    float *backward_output = output + lstm_param->batch_ * lstm_param->hidden_size_;
    float *backward_cell_state = cell_state + lstm_param->batch_ * lstm_param->hidden_size_;
    float *backward_hidden_state = hidden_state + lstm_param->batch_ * lstm_param->hidden_size_;
    for (int t = lstm_param->seq_len_ - 1; t >= 0; t--) {
      const float *input_ptr = input + t * lstm_param->input_step_;
      float *output_ptr = backward_output + t * lstm_param->output_step_;
      LstmStepUnit(output_ptr, input_ptr, backward_weight_i, backward_weight_h, backward_bias, backward_hidden_state,
                   backward_cell_state, gate_buffer, state_buffer, matmul_buffer, lstm_param);
    }
  }
}
