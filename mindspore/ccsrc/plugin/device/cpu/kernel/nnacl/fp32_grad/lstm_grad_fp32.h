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

#ifndef NNACL_FP32_GRAD_LSTM_GRAD_H_
#define NNACL_FP32_GRAD_LSTM_GRAD_H_

#include "nnacl/op_base.h"

typedef struct LstmGradParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  // shape correlative
  int input_size_;
  int hidden_size_;  // output_size
  int seq_len_;
  int batch_;
  // other parameter
  int output_step_;
  bool bidirectional_;
  float zoneout_cell_;
  float zoneout_hidden_;
  int input_row_align_;
  int input_col_align_;
  int state_row_align_;
  int state_col_align_;
  int has_bias_;
} LstmGradParameter;

#ifdef __cplusplus
extern "C" {
#endif

const int *getLstmOrderIOFG(void);

const int *getLstmOrderIFGO(void);

int GetRunWorkspaceSize(const LstmGradParameter *lstm_param);

size_t GetRunWorkspaceGemmOffset(const LstmGradParameter *lstm_param);

void LstmGradReorderDy(float *src, float *dst, LstmGradParameter *lstm_param);

void PackLstmWeightTranspose(float *dst, const float *src, int batch, int col, int row, int row_align,
                             const int *order);

void ReorderLstmWeights(float *dst, const float *src, int nof_martices, int col, int row, const int *order);

void LstmGradDoInputStep(const float *output_gate, float *cell_state, float *prev_cell_state, float *cell_gate,
                         float *input_gate, float *forget_gate, float *dY, float *dC, float *dH, float **dA, float *dX,
                         float *w, float *v, float *workspace, const LstmGradParameter *lstm_param);

void LstmGradDoWeightStep(float *input_t, float *prev_hidden_state, float *dA, float *dW, float *dV, float *dB,
                          float *workspace, const LstmGradParameter *lstm_param);
#ifdef __cplusplus
}
#endif
#endif  // NNACL_FP32_GRAD_LSTM_GRAD_H_
