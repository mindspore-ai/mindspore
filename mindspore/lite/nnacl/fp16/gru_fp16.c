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

void UpdateGruInputGateFp16(float16_t *gate_buffer, const float16_t *input, const float16_t *weight,
                            const float16_t *bias, int row, int deep, int col, int col_align, bool is_vec) {
  for (int i = 0; i < 3; i++) {
    const float16_t *weight_i = weight + deep * col * i;
    const float16_t *bias_i = bias + col_align * i;
    float16_t *gate = gate_buffer + row * col * i;
    LstmMatMulFp16(gate, input, weight_i, bias_i, row, deep, col, is_vec);
  }
}

void GruStepUnitFp16(float16_t *output, const float16_t *input, const float16_t *input_weight,
                     const float16_t *state_weight, const float16_t *bias, float16_t *hidden_state,
                     float16_t *gate_buffer, float16_t *matmul_buffer[2], const GruParameter *gru_param) {
  bool is_vec = gru_param->batch_ == 1;
  // input * weight
  if (is_vec) {
    UpdateGruInputGateFp16(gate_buffer, input, input_weight, bias, gru_param->batch_, gru_param->input_size_,
                           gru_param->hidden_size_, gru_param->col_align_, is_vec);
  } else {
    // pack input for matmul
    RowMajor2Col16MajorFp16(input, matmul_buffer[0], gru_param->batch_, gru_param->input_size_, false);
    UpdateGruInputGateFp16(gate_buffer, matmul_buffer[0], input_weight, bias, gru_param->batch_, gru_param->input_size_,
                           gru_param->hidden_size_, gru_param->col_align_, is_vec);
  }

  const float16_t *state_update_weight = state_weight;
  const float16_t *state_reset_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_;
  const float16_t *state_hidden_weight = state_weight + gru_param->hidden_size_ * gru_param->hidden_size_ * 2;
  float16_t *state_update_gate = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 3;
  float16_t *state_reset_gate = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 4;
  float16_t *state_hidden_buffer = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 5;
  const float16_t *state_update_bias = bias + gru_param->hidden_size_ * 3;
  const float16_t *state_reset_bias = bias + gru_param->hidden_size_ * 4;
  const float16_t *state_hidden_bias = bias + gru_param->hidden_size_ * 5;

  // state * weight
  if (is_vec) {
    LstmMatMulFp16(state_reset_gate, hidden_state, state_reset_weight, state_reset_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    LstmMatMulFp16(state_update_gate, hidden_state, state_update_weight, state_update_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    RowMajor2Col16MajorFp16(hidden_state, matmul_buffer[1], gru_param->batch_, gru_param->hidden_size_, false);
    LstmMatMulFp16(state_reset_gate, matmul_buffer[1], state_reset_weight, state_reset_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
    LstmMatMulFp16(state_update_gate, matmul_buffer[1], state_update_weight, state_update_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  }

  ElementAddFp16(gate_buffer, state_update_gate, gate_buffer, gru_param->batch_ * gru_param->hidden_size_ * 2);
  float16_t *update_gate = gate_buffer;
  float16_t *reset_gate = gate_buffer + gru_param->batch_ * gru_param->hidden_size_;
  float16_t *hidden_buffer = gate_buffer + gru_param->batch_ * gru_param->hidden_size_ * 2;

  // update reset_gate
  SigmoidFp16(reset_gate, reset_gate, gru_param->batch_ * gru_param->hidden_size_);

  // update update_gate
  SigmoidFp16(update_gate, update_gate, gru_param->batch_ * gru_param->hidden_size_);

  ElementMulFp16(hidden_state, reset_gate, reset_gate, gru_param->batch_ * gru_param->hidden_size_);
  if (is_vec) {
    LstmMatMulFp16(state_hidden_buffer, reset_gate, state_hidden_weight, state_hidden_bias, gru_param->batch_,
                   gru_param->hidden_size_, gru_param->hidden_size_, is_vec);
  } else {
    RowMajor2Col16MajorFp16(reset_gate, matmul_buffer[1], gru_param->batch_, gru_param->hidden_size_, false);
    LstmMatMulFp16(state_hidden_buffer, matmul_buffer[1], state_hidden_weight, state_hidden_bias, gru_param->batch_,
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

void GruFp16(float16_t *output, const float16_t *input, const float16_t *weight_g, const float16_t *weight_r,
             const float16_t *bias, float16_t *hidden_state, float16_t *gate_buffer, float16_t *matmul_buffer[2],
             int check_seq_len, const GruParameter *gru_param) {
  // forward
  for (int t = 0; t < check_seq_len; t++) {
    const float16_t *input_ptr = input + t * gru_param->input_step_;
    float16_t *output_ptr = output + t * gru_param->output_step_;
    GruStepUnitFp16(output_ptr, input_ptr, weight_g, weight_r, bias, hidden_state, gate_buffer, matmul_buffer,
                    gru_param);
  }
  // zero out extra fw outputs
  for (int t = check_seq_len; t < gru_param->seq_len_; t++) {
    float16_t *output_ptr = output + t * gru_param->output_step_;
    for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
      output_ptr[i] = 0.0f;
    }
  }

  // backward
  if (gru_param->bidirectional_) {
    const float16_t *backward_weight_g = weight_g + 3 * gru_param->col_align_ * gru_param->input_size_;
    const float16_t *backward_weight_r = weight_r + 3 * gru_param->col_align_ * gru_param->hidden_size_;
    const float16_t *backward_bias = bias + 6 * gru_param->hidden_size_;
    float16_t *backward_output = output + gru_param->batch_ * gru_param->hidden_size_;
    float16_t *backward_hidden_state = hidden_state + gru_param->batch_ * gru_param->hidden_size_;
    for (int t = check_seq_len - 1; t >= 0; t--) {
      const float16_t *input_ptr = input + t * gru_param->input_step_;
      float16_t *output_ptr = backward_output + t * gru_param->output_step_;
      GruStepUnitFp16(output_ptr, input_ptr, backward_weight_g, backward_weight_r, backward_bias, backward_hidden_state,
                      gate_buffer, matmul_buffer, gru_param);
    }
    // zero out extra bw outputs
    for (int t = gru_param->seq_len_ - 1; t >= check_seq_len; t--) {
      float16_t *output_ptr = backward_output + t * gru_param->output_step_;
      for (int i = 0; i < gru_param->batch_ * gru_param->hidden_size_; i++) {
        output_ptr[i] = 0.0f;
      }
    }
  }
}
