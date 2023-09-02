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
#include <float.h>
#include "nnacl/fp16/activation_fp16.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"

void PackLstmWeightFp32ToFp16(float16_t *dst, const float *src, int batch, int deep, int col, int col_align,
                              const int32_t *order) {
  for (int i = 0; i < batch; i++) {
    const float *src_batch = src + i * col * deep;
    float16_t *dst_batch = dst + (order == NULL ? i : order[i]) * col_align * deep;
#ifdef ENABLE_ARM64
    RowMajor2ColNMajorFp16(src_batch, dst_batch, col, deep, true);
#else
    RowMajor2Col8MajorFp16(src_batch, dst_batch, col, deep, true);
#endif
  }
}

void PackLstmWeightFp16(float16_t *dst, const float16_t *src, int batch, int deep, int col, int col_align,
                        const int32_t *order) {
  for (int i = 0; i < batch; i++) {
    const float16_t *src_batch = src + i * col * deep;
    float16_t *dst_batch = dst + (order == NULL ? i : order[i]) * col_align * deep;
#ifdef ENABLE_ARM64
    RowMajor2ColNMajorFp16(src_batch, dst_batch, col, deep, false);
#else
    RowMajor2Col8MajorFp16(src_batch, dst_batch, col, deep, false);
#endif
  }
}

void PackLstmBiasFp32ToFp16(float16_t *dst, const float *src, int batch, int col, int col_align, bool is_bidirectional,
                            const int32_t *order) {
  int unidirectional_batch = is_bidirectional ? batch / 2 : batch;
  for (int i = 0; i < unidirectional_batch; i++) {
    const float *src_batch = src + i * col;
    float16_t *dst_batch = dst + (order == NULL ? i : order[i]) * col_align;
    Float32ToFloat16(src_batch, dst_batch, col);
  }
  if (is_bidirectional) {
    const float *backward_src = src + batch * col;
    float16_t *backward_dst = dst + unidirectional_batch * col_align;
    for (int i = 0; i < unidirectional_batch; i++) {
      const float *backward_src_batch = backward_src + i * col;
      float16_t *backward_dst_batch = backward_dst + (order == NULL ? i : order[i]) * col_align;
      Float32ToFloat16(backward_src_batch, backward_dst_batch, col);
    }
  }
}

void PackLstmBiasFp16(float16_t *dst, const float16_t *src, int batch, int col, int col_align, bool is_bidirectional,
                      const int32_t *order) {
  int unidirectional_batch = is_bidirectional ? batch / 2 : batch;
  for (int i = 0; i < unidirectional_batch; i++) {
    const float16_t *src_batch = src + i * col;
    float16_t *dst_batch = dst + (order == NULL ? i : order[i]) * col_align;
    (void)memcpy(dst_batch, src_batch, col * sizeof(float16_t));
  }
  if (is_bidirectional) {
    const float16_t *backward_src = src + batch * col;
    float16_t *backward_dst = dst + unidirectional_batch * col_align;
    for (int i = 0; i < unidirectional_batch; i++) {
      const float16_t *backward_src_batch = backward_src + i * col;
      float16_t *backward_dst_batch = backward_dst + (order == NULL ? i : order[i]) * col_align;
      (void)memcpy(backward_dst_batch, backward_src_batch, col * sizeof(float16_t));
    }
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
    vout = MS_FMAQ_N_F16(vout, vin0, input1);
    vst1q_f16(output + index, vout);
  }
  for (; index < element_size; index++) {
    output[index] += input0[index] * input1;
  }
  return NNACL_OK;
}

void UpdateStateFp16(float16_t *cell_state, const float16_t *forget_gate, const float16_t *input_gate,
                     const float16_t *cell_gate, float16_t *state_buffer, int batch, int hidden_size,
                     float16_t zoneout) {
  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {  // zoneout * old_cell_state
    (void)memcpy(state_buffer, cell_state, batch * hidden_size * sizeof(float16_t));
    ElementOptMulFp16(state_buffer, &zoneout, state_buffer, batch * hidden_size, false);
  }

  ElementMulFp16(forget_gate, cell_state, cell_state, batch * hidden_size);
  ElementMulAccFp16(input_gate, cell_gate, cell_state, batch * hidden_size);

  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {  // (1 - zoneout) * new_cell_state
    ElementOptMulAccFp16(cell_state, 1 - zoneout, state_buffer, batch * hidden_size);
  }
}

void UpdateOutputFp16(float16_t *hidden_state, float16_t *output, const float16_t *cell_state, float16_t *output_gate,
                      const float16_t *weight_project, const float16_t *project_bias, float16_t *buffer[C7NUM],
                      const LstmParameter *lstm_param) {
  int batch = lstm_param->batch_;
  int hidden_size = lstm_param->hidden_size_;
  int output_size = lstm_param->output_size_;
  float16_t *state_buffer = buffer[C5NUM];
  float16_t *hidden_buffer = weight_project ? buffer[C3NUM] : hidden_state;
  float16_t zoneout = lstm_param->zoneout_hidden_;
  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {
    (void)memcpy(state_buffer, hidden_state, batch * output_size * sizeof(float16_t));
    ElementOptMulFp16(state_buffer, &zoneout, state_buffer, batch * output_size, false);
  }

  TanhFp16(cell_state, hidden_buffer, batch * hidden_size);
  ElementMulFp16(hidden_buffer, output_gate, hidden_buffer, batch * hidden_size);

  if (weight_project) {
    float16_t *left_matrix = hidden_buffer;
#ifdef ENABLE_ARM64
    if (batch >= C4NUM) {
      left_matrix = buffer[C6NUM];
      RowMajor2ColLadder12MajorFp16(hidden_buffer, left_matrix, batch, hidden_size);
    }
#else
    if (batch != 1) {
      left_matrix = buffer[C6NUM];
      RowMajor2Col16MajorFp16(hidden_buffer, left_matrix, batch, hidden_size, false);
    }
#endif
    LstmMatMulFp16(hidden_state, left_matrix, weight_project, project_bias, batch, hidden_size, output_size,
                   batch == 1);
  }
  if (!(zoneout >= -FLT_EPSILON && zoneout <= FLT_EPSILON)) {
    ElementOptMulAccFp16(hidden_state, 1 - zoneout, state_buffer, batch * output_size);
  }
  (void)memcpy(output, hidden_state, batch * output_size * sizeof(float16_t));
}

#ifdef ENABLE_ARM64
void LstmMatMulFp16(float16_t *c, const float16_t *a, const float16_t *b, const float16_t *bias, int row, int deep,
                    int col, bool is_vec) {
  MatmulFp16OptV2(a, b, c, bias, ActType_No, deep, row, col, col, OutType_Nhwc);
}
#else
void LstmMatMulFp16(float16_t *c, const float16_t *a, const float16_t *b, const float16_t *bias, int row, int deep,
                    int col, bool is_vec) {
  if (is_vec) {
    (void)memcpy(c, bias, col * sizeof(float16_t));
    MatMulAccFp16(c, a, b, row, col, deep);
  } else {
    MatMulFp16(a, b, c, bias, ActType_No, deep, row, col, col, OutType_Nhwc);
  }
}
#endif

void UpdateLstmGateFp16(float16_t *gate_buffer, const float16_t *input, const float16_t *weight, const float16_t *bias,
                        int row, int deep, int col, int col_align, bool is_vec) {
  for (int i = 0; i < 4; i++) {
    const float16_t *weight_i = weight + deep * col_align * i;
    const float16_t *bias_i = bias + col_align * i;
    float16_t *gate = gate_buffer + row * col * i;
    LstmMatMulFp16(gate, input, weight_i, bias_i, row, deep, col, is_vec);
  }
}

void LstmStepUnitFp16(float16_t *output, float16_t *input_gate, float16_t *forget_gate, float16_t *cell_gate,
                      float16_t *output_gate, const float16_t *state_weight, const float16_t *state_bias,
                      const float16_t *weight_project, const float16_t *project_bias, float16_t *hidden_state,
                      float16_t *cell_state, float16_t *buffer[C7NUM], const LstmParameter *lstm_param) {
  float16_t *packed_state = buffer[C2NUM];
  float16_t *state_gate = buffer[C3NUM];
  float16_t *cell_buffer = buffer[C4NUM];
  float16_t *hidden_buffer = buffer[C5NUM];
  bool is_vec = lstm_param->batch_ == 1;
#ifdef ENABLE_ARM64
  if (lstm_param->batch_ <= C3NUM) {
    UpdateLstmGateFp16(state_gate, hidden_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                       lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec);
  } else {
    RowMajor2ColLadder12MajorFp16(hidden_state, packed_state, lstm_param->batch_, lstm_param->output_size_);
    UpdateLstmGateFp16(state_gate, packed_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                       lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec);
  }
#else
  if (is_vec) {
    UpdateLstmGateFp16(state_gate, hidden_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                       lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec);
  } else {
    RowMajor2Col16MajorFp16(hidden_state, packed_state, lstm_param->batch_, lstm_param->output_size_, false);
    UpdateLstmGateFp16(state_gate, packed_state, state_weight, state_bias, lstm_param->batch_, lstm_param->output_size_,
                       lstm_param->hidden_size_, lstm_param->state_col_align_, is_vec);
  }
#endif
  ElementAddFp16(input_gate, state_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_);
  ElementAddFp16(forget_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 2, forget_gate,
                 lstm_param->batch_ * lstm_param->hidden_size_);
  ElementAddFp16(cell_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_ * 3, cell_gate,
                 lstm_param->batch_ * lstm_param->hidden_size_);
  ElementAddFp16(output_gate, state_gate + lstm_param->batch_ * lstm_param->hidden_size_, output_gate,
                 lstm_param->batch_ * lstm_param->hidden_size_);

  // update input_gate
  SigmoidFp16(input_gate, input_gate, lstm_param->batch_ * lstm_param->hidden_size_);

  // update forget_gate
  SigmoidFp16(forget_gate, forget_gate, lstm_param->batch_ * lstm_param->hidden_size_);

  // update cell_gate
  TanhFp16(cell_gate, cell_gate, lstm_param->batch_ * lstm_param->hidden_size_);
  // update cell state
  UpdateStateFp16(cell_state, forget_gate, input_gate, cell_gate, cell_buffer, lstm_param->batch_,
                  lstm_param->hidden_size_, lstm_param->zoneout_cell_);

  // update output_gate
  SigmoidFp16(output_gate, output_gate, lstm_param->batch_ * lstm_param->hidden_size_);
  // update output
  UpdateOutputFp16(hidden_state, output, cell_state, output_gate, weight_project, project_bias, buffer, lstm_param);

  if (!(lstm_param->zoneout_cell_ >= -FLT_EPSILON && lstm_param->zoneout_cell_ <= FLT_EPSILON)) {
    (void)memcpy(cell_state, cell_buffer, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float16_t));
  }

  if (!(lstm_param->zoneout_hidden_ >= -FLT_EPSILON && lstm_param->zoneout_hidden_ <= FLT_EPSILON)) {
    (void)memcpy(hidden_state, hidden_buffer, lstm_param->batch_ * lstm_param->output_size_ * sizeof(float16_t));
  }
}

void LstmGateCompute(float16_t *gate, const float16_t *input, const float16_t *weight_i, const float16_t *input_bias,
                     const LstmParameter *lstm_param) {
  int row_input = lstm_param->seq_len_ * lstm_param->batch_;
  for (int i = 0; i < C4NUM; i++) {
    const float16_t *weight_loop = weight_i + lstm_param->input_size_ * lstm_param->input_col_align_ * i;
    const float16_t *bias_loop = input_bias + lstm_param->input_col_align_ * i;
    float16_t *gate_loop = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * i;
#ifdef ENABLE_ARM64
    MatmulFp16OptV2(input, weight_loop, gate_loop, bias_loop, ActType_No, lstm_param->input_size_, row_input,
                    lstm_param->hidden_size_, lstm_param->hidden_size_, OutType_Nhwc);
#else
    MatMulFp16(input, weight_loop, gate_loop, bias_loop, ActType_No, lstm_param->input_size_, row_input,
               lstm_param->hidden_size_, lstm_param->hidden_size_, OutType_Nhwc);
#endif
  }
}

void LstmUnidirectionalFp16(float16_t *output, const float16_t *packed_input, const float16_t *weight_i,
                            const float16_t *weight_h, const float16_t *input_bias, const float16_t *state_bias,
                            const float16_t *weight_project, const float16_t *project_bias, float16_t *hidden_state,
                            float16_t *cell_state, float16_t *buffer[C7NUM], const LstmParameter *lstm_param,
                            bool is_backward) {
  float16_t *gate = buffer[1];
  LstmGateCompute(gate, packed_input, weight_i, input_bias, lstm_param);

  float16_t *input_gate = gate;
  float16_t *forget_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 2;
  float16_t *cell_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_ * 3;
  float16_t *output_gate = gate + lstm_param->seq_len_ * lstm_param->batch_ * lstm_param->hidden_size_;
  for (int t = 0; t < lstm_param->seq_len_; t++) {
    int real_t = is_backward ? lstm_param->seq_len_ - t - 1 : t;
    float16_t *input_gate_t = input_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float16_t *forget_gate_t = forget_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float16_t *cell_gate_t = cell_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float16_t *output_gate_t = output_gate + lstm_param->batch_ * lstm_param->hidden_size_ * real_t;
    float16_t *output_ptr = output + real_t * lstm_param->output_step_;
    LstmStepUnitFp16(output_ptr, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, weight_h, state_bias,
                     weight_project, project_bias, hidden_state, cell_state, buffer, lstm_param);
  }
}

void LstmFp16(float16_t *output, const float16_t *input, const float16_t *weight_i, const float16_t *weight_h,
              const float16_t *input_bias, const float16_t *state_bias, const float16_t *weight_project,
              const float16_t *project_bias, float16_t *hidden_state, float16_t *cell_state, float16_t *buffer[C7NUM],
              const LstmParameter *lstm_param) {
  // forward
#ifdef ENABLE_ARM64
  const float16_t *packed_input = input;
  if (lstm_param->batch_ * lstm_param->seq_len_ >= C4NUM) {
    float16_t *temp_input = buffer[0];
    RowMajor2ColLadder12MajorFp16(input, temp_input, lstm_param->seq_len_ * lstm_param->batch_,
                                  lstm_param->input_size_);
    packed_input = temp_input;
  }
#else
  float16_t *packed_input = buffer[0];
  RowMajor2Col16MajorFp16(input, packed_input, lstm_param->seq_len_ * lstm_param->batch_, lstm_param->input_size_,
                          false);
#endif
  LstmUnidirectionalFp16(output, packed_input, weight_i, weight_h, input_bias, state_bias, weight_project, project_bias,
                         hidden_state, cell_state, buffer, lstm_param, false);

  // backward
  if (lstm_param->bidirectional_) {
    const float16_t *backward_weight_i = weight_i + 4 * lstm_param->input_col_align_ * lstm_param->input_size_;
    const float16_t *backward_weight_h = weight_h + 4 * lstm_param->state_col_align_ * lstm_param->output_size_;
    const float16_t *backward_input_bias = input_bias + 4 * lstm_param->input_col_align_;
    const float16_t *backward_state_bias = state_bias + 4 * lstm_param->state_col_align_;
    const float16_t *backward_weight_project =
      weight_project ? weight_project + lstm_param->hidden_size_ * lstm_param->proj_col_align_ : NULL;
    float16_t *backward_output = output + lstm_param->batch_ * lstm_param->output_size_;
    float16_t *backward_cell_state = cell_state + lstm_param->batch_ * lstm_param->hidden_size_;
    float16_t *backward_hidden_state = hidden_state + lstm_param->batch_ * lstm_param->output_size_;

    LstmUnidirectionalFp16(backward_output, packed_input, backward_weight_i, backward_weight_h, backward_input_bias,
                           backward_state_bias, backward_weight_project, project_bias, backward_hidden_state,
                           backward_cell_state, buffer, lstm_param, true);
  }
}
