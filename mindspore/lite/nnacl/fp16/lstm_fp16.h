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

#ifndef MINDSPORE_LITE_NNACL_FP16_LSTM_H_
#define MINDSPORE_LITE_NNACL_FP16_LSTM_H_

#include "nnacl/lstm_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif
void PackLstmWeightFp32ToFp16(float16_t *dst, const float *src, int batch, int deep, int col, int col_align);

void PackLstmWeightFp16(float16_t *dst, const float16_t *src, int batch, int deep, int col, int col_align);

void PackLstmBiasFp32ToFp16(float16_t *dst, const float *src, int batch, int col, int col_align, bool is_bidirectional);

void PackLstmBiasFp16(float16_t *dst, const float16_t *src, int batch, int col, int col_align, bool is_bidirectional);

void LstmMatMulFp16(float16_t *c, const float16_t *a, const float16_t *b, const float16_t *bias, int row, int deep,
                    int col, bool is_vec);

void MatMulAccFp16(float16_t *output, const float16_t *input, const float16_t *weight, int rows, int cols,
                   int inner_size);

void ElementMulAccFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementOptMulAccFp16(const float16_t *input0, const float16_t input1, float16_t *output, const int element_size);

void LstmFp16(float16_t *output, const float16_t *input, const float16_t *weight_i, const float16_t *weight_h,
              const float16_t *input_bias, const float16_t *state_bias, float16_t *hidden_state, float16_t *cell_state,
              float16_t *buffer[6], const LstmParameter *lstm_param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_LSTM_H_
