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

void InitGruGateFp16(float16_t *gate_buffer, const float16_t *bias, const GruParameter *gru_parm) {
  int gate_offest = 0;
  for (int l = 0; l < 3; l++) {
    int batch_offest = gate_offest;
    int bias_offest = l * gru_parm->hidden_size_;
    for (int b = 0; b < gru_parm->batch_; b++) {
      memcpy(gate_buffer + batch_offest, bias + bias_offest, gru_parm->hidden_size_ * sizeof(float16_t));
      batch_offest += gru_parm->hidden_size_;
    }
    gate_offest += gru_parm->batch_ * gru_parm->hidden_size_;
  }
}

void GruStepUnitFp16(float16_t *output, const float16_t *input, const float16_t *input_reset_weight,
                     const float16_t *input_update_weight, const float16_t *input_hidden_weight,
                     const float16_t *state_reset_weight, const float16_t *state_update_weight,
                     const float16_t *state_hidden_weight, const float16_t *bias, float16_t *hidden_state,
                     float16_t *gate_buffer, const GruParameter *gru_parm) {
  InitGruGateFp16(gate_buffer, bias, gru_parm);

  float16_t *update_gate = gate_buffer;
  float16_t *reset_gate = gate_buffer + gru_parm->batch_ * gru_parm->hidden_size_;
  float16_t *hidden_buffer = gate_buffer + gru_parm->batch_ * gru_parm->hidden_size_ * 2;

  // input * weight
  MatMulAccFp16(reset_gate, input, input_reset_weight, gru_parm->batch_, gru_parm->hidden_size_, gru_parm->input_size_);
  MatMulAccFp16(update_gate, input, input_update_weight, gru_parm->batch_, gru_parm->hidden_size_,
                gru_parm->input_size_);
  MatMulAccFp16(hidden_buffer, input, input_hidden_weight, gru_parm->batch_, gru_parm->hidden_size_,
                gru_parm->input_size_);

  // state * weight
  MatMulAccFp16(reset_gate, hidden_state, state_reset_weight, gru_parm->batch_, gru_parm->hidden_size_,
                gru_parm->hidden_size_);
  MatMulAccFp16(update_gate, hidden_state, state_update_weight, gru_parm->batch_, gru_parm->hidden_size_,
                gru_parm->hidden_size_);

  // update reset_gate
  SigmoidFp16(reset_gate, reset_gate, gru_parm->batch_ * gru_parm->hidden_size_);

  // update update_gate
  SigmoidFp16(update_gate, update_gate, gru_parm->batch_ * gru_parm->hidden_size_);

  ElementMulFp16(hidden_state, reset_gate, reset_gate, gru_parm->batch_ * gru_parm->hidden_size_);
  MatMulAccFp16(hidden_buffer, reset_gate, state_hidden_weight, gru_parm->batch_, gru_parm->hidden_size_,
                gru_parm->hidden_size_);

  TanhFp16(hidden_buffer, hidden_buffer, gru_parm->batch_ * gru_parm->hidden_size_);

  ElementMulFp16(update_gate, hidden_state, hidden_state, gru_parm->batch_ * gru_parm->hidden_size_);

  ArithmeticParameter parameter;
  parameter.in_elements_num0_ = 1;
  parameter.in_elements_num1_ = gru_parm->batch_ * gru_parm->hidden_size_;
  float16_t one = 1.0f;
  ElementOptSubFp16(&one, update_gate, update_gate, gru_parm->batch_ * gru_parm->hidden_size_, &parameter);

  ElementMulAccFp16(update_gate, hidden_buffer, hidden_state, gru_parm->batch_ * gru_parm->hidden_size_);

  memcpy(output, hidden_state, gru_parm->batch_ * gru_parm->hidden_size_ * sizeof(float16_t));
}

void GruFp16(float16_t *output, const float16_t *input, const float16_t *weight_g, const float16_t *weight_r,
             const float16_t *bias, float16_t *hidden_state, float16_t *gate_buffer, int check_seq_len,
             const GruParameter *gru_parm) {
  // forward
  const float16_t *input_update_weight = weight_g;
  const float16_t *input_reset_weight = weight_g + gru_parm->input_size_ * gru_parm->hidden_size_;
  const float16_t *input_hidden_weight = weight_g + gru_parm->input_size_ * gru_parm->hidden_size_ * 2;

  const float16_t *state_update_weight = weight_r;
  const float16_t *state_reset_weight = weight_r + gru_parm->hidden_size_ * gru_parm->hidden_size_;
  const float16_t *state_hidden_weight = weight_r + gru_parm->hidden_size_ * gru_parm->hidden_size_ * 2;

  for (int t = 0; t < check_seq_len; t++) {
    const float16_t *input_ptr = input + t * gru_parm->input_step_;
    float16_t *output_ptr = output + t * gru_parm->output_step_;
    GruStepUnitFp16(output_ptr, input_ptr, input_reset_weight, input_update_weight, input_hidden_weight,
                    state_reset_weight, state_update_weight, state_hidden_weight, bias, hidden_state, gate_buffer,
                    gru_parm);
  }
  // zero out extra fw outputs
  for (int t = check_seq_len; t < gru_parm->seq_len_; t++) {
    float16_t *output_ptr = output + t * gru_parm->output_step_;
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

    float16_t *backward_output = output + gru_parm->batch_ * gru_parm->hidden_size_;
    const float16_t *backward_bias = bias + 3 * gru_parm->hidden_size_;
    float16_t *backward_hidden_state = hidden_state + gru_parm->batch_ * gru_parm->hidden_size_;
    for (int t = check_seq_len - 1; t >= 0; t--) {
      const float16_t *input_ptr = input + t * gru_parm->input_step_;
      float16_t *output_ptr = backward_output + t * gru_parm->output_step_;
      GruStepUnitFp16(output_ptr, input_ptr, input_reset_weight, input_update_weight, input_hidden_weight,
                      state_reset_weight, state_update_weight, state_hidden_weight, backward_bias,
                      backward_hidden_state, gate_buffer, gru_parm);
    }
    // zero out extra bw outputs
    for (int t = gru_parm->seq_len_ - 1; t >= check_seq_len; t--) {
      float16_t *output_ptr = backward_output + t * gru_parm->output_step_;
      for (int i = 0; i < gru_parm->batch_ * gru_parm->hidden_size_; i++) {
        output_ptr[i] = 0.0f;
      }
    }
  }
}
