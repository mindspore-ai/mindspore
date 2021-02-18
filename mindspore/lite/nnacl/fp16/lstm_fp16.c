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

#include "nnacl/fp16/lstm_fp16.h"
#include <string.h>
#include "nnacl/fp16/activation_fp16.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"

void PackLstmWeightFp32ToFp16(float16_t *dst, const float *src, int batch, int deep, int col, int col_align) {
  for (int i = 0; i < batch; i++) {
    const float *src_batch = src + i * col * deep;
    float16_t *dst_batch = dst + i * col_align * deep;
    RowMajor2Col8MajorFp16(src_batch, dst_batch, col, deep, true);
  }
}

void PackLstmWeightFp16(float16_t *dst, const float16_t *src, int batch, int deep, int col, int col_align) {
  for (int i = 0; i < batch; i++) {
    const float16_t *src_batch = src + i * col * deep;
    float16_t *dst_batch = dst + i * col_align * deep;
    RowMajor2Col8MajorFp16(src_batch, dst_batch, col, deep, false);
  }
}

// input: [row, inner_size]; weight: [col, inner_size]; output: [row, col]
void MatMulAccFp16(float16_t *output, const float16_t *input, const float16_t *weight, int rows, int cols,
                   int inner_size) {
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float16_t res = 0;
      const float16_t *input_col = input + r * inner_size;
      const float16_t *weight_col = weight + c * inner_size;
      int index = 0;
      float16x8_t out = vdupq_n_f16(0.0f);
      for (; index <= inner_size - 8; index += 8) {
        float16x8_t in_0 = vld1q_f16(input_col + index);
        float16x8_t in_1 = vld1q_f16(weight_col + index);
        out = vfmaq_f16(out, in_1, in_0);
      }
      float16x4_t add2 = vadd_f16(vget_low_f16(out), vget_high_f16(out));
      float16x4_t add4 = vpadd_f16(add2, add2);
      float16x4_t add8 = vpadd_f16(add4, add4);
      res += vget_lane_f16(add8, 0);
      for (; index < inner_size; index++) {
        res += input_col[index] * weight_col[index];
      }
      output[r * cols + c] += res;
    }
  }
}

void ElementMulAccFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size) {
  int index = 0;
  for (; index <= element_size - 8; index += 8) {
    float16x8_t in_0 = vld1q_f16(input0 + index);
    float16x8_t in_1 = vld1q_f16(input1 + index);
    float16x8_t out = vld1q_f16(output + index);
    out = vfmaq_f16(out, in_1, in_0);
    vst1q_f16(output + index, out);
  }
  for (; index < element_size; index++) {
    output[index] += input0[index] * input1[index];
  }
}

int ElementOptMulAccFp16(const float16_t *input0, const float16_t input1, float16_t *output, const int element_size) {
  int index = 0;
  for (; index <= element_size - 8; index += 8) {
    float16x8_t vin0 = vld1q_f16(input0 + index);
    float16x8_t vout = vld1q_f16(output + index);
    vout = vfmaq_n_f16(vout, vin0, input1);
    vst1q_f16(output + index, vout);
  }
  for (; index < element_size; index++) {
    output[index] += input0[index] * input1;
  }
  return NNACL_OK;
}

void UpdataStateFp16(float16_t *cell_state, float16_t *forget_gate, const float16_t *input_gate,
                     const float16_t *cell_gate, float16_t *state_buffer, int batch, int hidden_size,
                     float16_t smooth) {
  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {  // smooth * old_cell_state
    memcpy(state_buffer, cell_state, batch * hidden_size * sizeof(float16_t));
    ArithmeticParameter parameter;
    parameter.in_elements_num0_ = batch * hidden_size;
    parameter.in_elements_num1_ = 1;
    ElementOptMulFp16(state_buffer, &smooth, state_buffer, batch * hidden_size, &parameter);
  }

  ElementMulFp16(forget_gate, cell_state, cell_state, batch * hidden_size);
  ElementMulAccFp16(input_gate, cell_gate, cell_state, batch * hidden_size);

  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {  // (1 - smooth) * new_cell_state
    ElementOptMulAccFp16(cell_state, 1 - smooth, state_buffer, batch * hidden_size);
  }
}

void UpdataOutputFp16(const float16_t *cell_state, float16_t *output_gate, float16_t *hidden_state,
                      float16_t *state_buffer_in, int batch, int hidden_size, float16_t smooth) {
  float16_t *state_buffer = state_buffer_in + batch * hidden_size;
  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {
    memcpy(state_buffer, hidden_state, batch * hidden_size * sizeof(float16_t));
    ArithmeticParameter parameter;
    parameter.in_elements_num0_ = batch * hidden_size;
    parameter.in_elements_num1_ = 1;
    ElementOptMulFp16(state_buffer, &smooth, state_buffer, batch * hidden_size, &parameter);
  }

  TanhFp16(cell_state, hidden_state, batch * hidden_size);
  ElementMulFp16(hidden_state, output_gate, hidden_state, batch * hidden_size);

  if (!(smooth >= -FLT_EPSILON && smooth <= FLT_EPSILON)) {
    ElementOptMulAccFp16(hidden_state, 1 - smooth, state_buffer, batch * hidden_size);
  }
}

void LstmMatMulFp16(float16_t *c, const float16_t *a, const float16_t *b, const float16_t *bias, int row, int deep,
                    int col, bool is_vec) {
  if (is_vec) {
    memcpy(c, bias, col * sizeof(float16_t));
    MatMulAccFp16(c, a, b, row, col, deep);
  } else {
    MatMulFp16(a, b, c, bias, ActType_No, deep, row, col, col, OutType_Nhwc);
  }
}

void UpdateLstmGateFp16(float16_t *gate_buffer, const float16_t *input, const float16_t *weight, const float16_t *bias,
                        int row, int deep, int col, int col_align, bool is_vec) {
  for (int i = 0; i < 4; i++) {
    const float16_t *weight_i = weight + deep * col * i;
    const float16_t *bias_i = bias + col_align * i;
    float16_t *gate = gate_buffer + row * col * i;
    LstmMatMulFp16(gate, input, weight_i, bias_i, row, deep, col, is_vec);
  }
}

void LstmStepUnitFp16(float16_t *output, const float16_t *input, const float16_t *input_weight,
                      const float16_t *state_weight, const float16_t *bias, float16_t *hidden_state,
                      float16_t *cell_state, float16_t *gate_buffer, float16_t *state_buffer,
                      float16_t *matmul_buffer[2], const LstmParameter *lstm_param) {
  bool is_vec = lstm_param->batch_ == 1;
  // input * weight
  if (is_vec) {
    UpdateLstmGateFp16(gate_buffer, input, input_weight, bias, lstm_param->batch_, lstm_param->input_size_,
                       lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  } else {
    // pack input for matmul
    RowMajor2Col16MajorFp16(input, matmul_buffer[0], lstm_param->batch_, lstm_param->input_size_, false);
    UpdateLstmGateFp16(gate_buffer, matmul_buffer[0], input_weight, bias, lstm_param->batch_, lstm_param->input_size_,
                       lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  }

  // state * weight
  float16_t *state_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_ * 4;
  const float16_t *state_bias = bias + lstm_param->col_align_ * 4;
  if (is_vec) {
    UpdateLstmGateFp16(state_gate, hidden_state, state_weight, state_bias, lstm_param->batch_, lstm_param->hidden_size_,
                       lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  } else {
    // pack state for matmul
    RowMajor2Col16MajorFp16(hidden_state, matmul_buffer[1], lstm_param->batch_, lstm_param->hidden_size_, false);
    UpdateLstmGateFp16(state_gate, matmul_buffer[1], state_weight, state_bias, lstm_param->batch_,
                       lstm_param->hidden_size_, lstm_param->hidden_size_, lstm_param->col_align_, is_vec);
  }
  ElementAddFp16(gate_buffer, state_gate, gate_buffer, 4 * lstm_param->batch_ * lstm_param->hidden_size_);

  float16_t *input_gate = gate_buffer;
  float16_t *forget_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_ * 2;
  float16_t *cell_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_ * 3;
  float16_t *output_gate = gate_buffer + lstm_param->batch_ * lstm_param->hidden_size_;
  // update input_gate
  SigmoidFp16(input_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_);

  // update forget_gate
  SigmoidFp16(forget_gate, forget_gate, lstm_param->batch_ * lstm_param->hidden_size_);

  // update cell_gate
  TanhFp16(cell_gate, cell_gate, lstm_param->batch_ * lstm_param->hidden_size_);
  // update cell state
  UpdataStateFp16(cell_state, forget_gate, input_gate, cell_gate, state_buffer, lstm_param->batch_,
                  lstm_param->hidden_size_, lstm_param->smooth_);

  // update output_gate
  SigmoidFp16(output_gate, output_gate, lstm_param->batch_ * lstm_param->hidden_size_);
  // update output
  UpdataOutputFp16(cell_state, output_gate, hidden_state, state_buffer, lstm_param->batch_, lstm_param->hidden_size_,
                   lstm_param->smooth_);
  memcpy(output, hidden_state, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float16_t));

  if (!(lstm_param->smooth_ >= -FLT_EPSILON && lstm_param->smooth_ <= FLT_EPSILON)) {
    memcpy(cell_state, state_buffer, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float16_t));
    memcpy(hidden_state, state_buffer + lstm_param->batch_ * lstm_param->hidden_size_,
           lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float16_t));
  }
}

void LstmFp16(float16_t *output, const float16_t *input, const float16_t *weight_i, const float16_t *weight_h,
              const float16_t *bias, float16_t *hidden_state, float16_t *cell_state, float16_t *gate_buffer,
              float16_t *state_buffer, float16_t *matmul_buffer[2], const LstmParameter *lstm_param) {
  // forward
  for (int t = 0; t < lstm_param->seq_len_; t++) {
    const float16_t *input_ptr = input + t * lstm_param->input_step_;
    float16_t *output_ptr = output + t * lstm_param->output_step_;
    LstmStepUnitFp16(output_ptr, input_ptr, weight_i, weight_h, bias, hidden_state, cell_state, gate_buffer,
                     state_buffer, matmul_buffer, lstm_param);
  }

  // backward
  if (lstm_param->bidirectional_) {
    const float16_t *backward_weight_i = weight_i + 4 * lstm_param->col_align_ * lstm_param->input_size_;
    const float16_t *backward_weight_h = weight_h + 4 * lstm_param->col_align_ * lstm_param->hidden_size_;
    const float16_t *backward_bias = bias + 8 * lstm_param->col_align_;
    float16_t *backward_output = output + lstm_param->batch_ * lstm_param->hidden_size_;
    float16_t *backward_cell_state = cell_state + lstm_param->batch_ * lstm_param->hidden_size_;
    float16_t *backward_hidden_state = hidden_state + lstm_param->batch_ * lstm_param->hidden_size_;
    for (int t = lstm_param->seq_len_ - 1; t >= 0; t--) {
      const float16_t *input_ptr = input + t * lstm_param->input_step_;
      float16_t *output_ptr = backward_output + t * lstm_param->output_step_;
      LstmStepUnitFp16(output_ptr, input_ptr, backward_weight_i, backward_weight_h, backward_bias,
                       backward_hidden_state, backward_cell_state, gate_buffer, state_buffer, matmul_buffer,
                       lstm_param);
    }
  }
}
