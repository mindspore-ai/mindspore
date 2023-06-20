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

#ifndef MINDSPORE_NNACL_FP32_LSTM_H_
#define MINDSPORE_NNACL_FP32_LSTM_H_

#include "nnacl/lstm_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif
void PackLstmWeight(float *dst, const float *src, int batch, int deep, int col, int col_align, const int32_t *order);

void PackLstmWeightWithStride(float *dst, const float *src, int batch, int deep, int col, int col_align,
                              bool is_bidirectional, int stride, const int32_t *order);

void PackLstmBias(float *dst, const float *src, int batch, int col, int col_align, bool is_bidirectional,
                  const int32_t *order);

void PackLstmBiasWithStride(float *dst, const float *src, int batch, int col, int col_align, bool is_bidirectional,
                            int b_stride, const int32_t *order);

void PackLstmInput(const float *src, float *dst, int row, int deep);

void LstmMatMul(float *c, const float *a, const float *b, const float *bias, int row, int deep, int col, int col_align,
                bool is_vec, float *packed_ptr);

void ElementMulAcc(const float *input0, const float *input1, float *output, int element_size);

int ElementOptMulAcc(const float *input0, const float input1, float *output, const int element_size);

void LstmStepUnit(float *output, float *input_gate, float *forget_gate, float *cell_gate, float *output_gate,
                  const float *state_weight, const float *state_bias, const float *weight_project, float *hidden_state,
                  float *cell_state, float *buffer[C8NUM], const LstmParameter *lstm_param);

void Lstm(float *output, const float *input, const float *weight_i, const float *weight_h, const float *input_bias,
          const float *state_bias, float *hidden_state, float *cell_state, float *buffer[C9NUM],
          const LstmParameter *lstm_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_LSTM_H_
