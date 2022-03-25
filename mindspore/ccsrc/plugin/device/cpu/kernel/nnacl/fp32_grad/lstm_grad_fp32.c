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

#include "nnacl/fp32_grad/lstm_grad_fp32.h"
#include <string.h>
#include <float.h>
#include "nnacl/lstm_parameter.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32_grad/gemm.h"
#include "nnacl/fp32/lstm_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/nnacl_utils.h"

static const int num_of_gates = 4;
static const int no_of_temp_matrices_sized_output_step = 5;

static inline float *AllocteFromScrachPad(float **scrach_pad, int size) {
  float *buffer = *scrach_pad;
  *scrach_pad += size;
  return buffer;
}

static const int weights_order_IOFG[2 * 4] = {0, 3, 1, 2, 4, 7, 5, 6};  // IOFG order to IFGO order
static const int weights_order_IFGO[2 * 4] = {0, 2, 3, 1, 4, 6, 7, 5};  // IFGO order to IOFG order

const int *getLstmOrderIOFG(void) { return weights_order_IOFG; }

const int *getLstmOrderIFGO(void) { return weights_order_IFGO; }

void PackLstmWeightTranspose(float *dst, const float *src, int batch, int col, int row, int row_align,
                             const int *order) {
  for (int i = 0; i < batch; i++) {
    const float *src_batch = src + i * col * row;
    float *dst_batch = dst + ((order == NULL) ? i : order[i]) * col * row_align;
#ifdef ENABLE_AVX
    RowMajor2Row16Major(src_batch, dst_batch, row, col);
#elif defined(ENABLE_ARM32)
    RowMajor2Row4Major(src_batch, dst_batch, row, col);
#else
    RowMajor2Row8Major(src_batch, dst_batch, row, col);
#endif
  }
}

void ReorderLstmWeights(float *dst, const float *src, int nof_martices, int col, int row, const int *order) {
  int matrix_size = col * row;
  for (int i = 0; i < nof_martices; i++) {
    const float *src_block = src + i * matrix_size;
    float *dst_block = dst + ((order == NULL) ? i : order[i]) * matrix_size;
    memcpy(dst_block, src_block, matrix_size * sizeof(float));
  }
}

void sumCols(int m, int n, int stride, float *inMat, float *outMat, bool accumulate) {
  for (int idn = 0; idn < n; idn++) {
    float *col = inMat + idn;
    if (!accumulate) {
      *outMat = 0;
    }
    for (int idm = 0; idm < m; idm++) {
      *outMat += *col;
      col += stride;
    }
    outMat++;
  }
}

int GetGemmMatMullWorkspace(int batch, int input_size, int hidden_size) {
  int workspace_size, temp;
  // if the appropriate GemmMatNul use beta>0 matSizeTotal must have col as last parameter.
  workspace_size = MatSizeTotal(batch, input_size, hidden_size, input_size);
  temp = MatSizeTotal(batch, hidden_size, hidden_size, hidden_size);
  workspace_size = (temp > workspace_size) ? temp : workspace_size;
  temp = MatSizeTotal(hidden_size, input_size, batch, input_size);
  workspace_size = (temp > workspace_size) ? temp : workspace_size;
  temp = MatSizeTotal(hidden_size, hidden_size, batch, hidden_size);
  workspace_size = (temp > workspace_size) ? temp : workspace_size;
  return workspace_size;
}

int GetRunWorkspaceSize(const LstmGradParameter *lstm_param) {
  int time_stamp_len = lstm_param->batch_ * lstm_param->hidden_size_;
  int workspace_size = no_of_temp_matrices_sized_output_step * time_stamp_len;
  workspace_size += GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_);
  return workspace_size;
}

size_t GetRunWorkspaceGemmOffset(const LstmGradParameter *lstm_param) {
  int time_stamp_len = lstm_param->batch_ * lstm_param->hidden_size_;
  return no_of_temp_matrices_sized_output_step * time_stamp_len;
}

void LstmGradReorderDy(float *src, float *dst, LstmGradParameter *lstm_param) {
  int dir_mult = lstm_param->bidirectional_ ? C2NUM : C1NUM;
  for (int b = 0; b < lstm_param->batch_; b++) {
    int batch_offset = b * dir_mult * lstm_param->hidden_size_;
    float *dy = src + batch_offset;
    memcpy(dst + b * lstm_param->hidden_size_, dy, lstm_param->hidden_size_ * sizeof(float));
  }
}

void LstmGradDoInputStep(const float *output_gate, float *cell_state, float *prev_cell_state, float *cell_gate,
                         float *input_gate, float *forget_gate, float *dY, float *dC, float *dH, float **dA, float *dX,
                         float *w, float *v, float *workspace, const LstmGradParameter *lstm_param) {
  float *scratchPad = workspace;

  int seq_len = lstm_param->batch_ * lstm_param->hidden_size_;
  float *temp0 = AllocteFromScrachPad(&scratchPad, seq_len);
  float *temp1 = AllocteFromScrachPad(&scratchPad, seq_len);
  float *temp2 = AllocteFromScrachPad(&scratchPad, seq_len);
  float *temp3 = AllocteFromScrachPad(&scratchPad, seq_len);
  float *temp4 = AllocteFromScrachPad(&scratchPad, seq_len);

  // Accumulate gradients into dH
  ElementAdd(dH, dY, dH, seq_len);

  ElementMul(dH, output_gate, temp1, seq_len);
  Tanh(cell_state, seq_len, temp0);
  ElementMul(temp0, temp0, temp2, seq_len);
  ElementMul(temp1, temp2, temp4, seq_len);
  ElementSub(temp1, temp4, temp1, seq_len);
  ElementAdd(dC, temp1, dC, seq_len);

  // calculate dI, dO, dF and dG
  float *dI = temp1;  // dI = dC_{t} * G
  ElementMul(dC, cell_gate, dI, seq_len);
  float *dO = temp2;  // dO = dH * Tanh(C_{t})
  ElementMul(dH, temp0, dO, seq_len);
  float *dF = temp3;  // dF = dC_{t} * C_{t-1}
  ElementMul(dC, prev_cell_state, dF, seq_len);
  float *dG = temp4;  // dG = dC_{t} * I
  ElementMul(dC, input_gate, dG, seq_len);

  // dAi = dI * I * (1 - I)
  float *dAi = temp1;
  *dA = dAi;
  ElementMul(dI, input_gate, dAi, seq_len);
  ElementMul(dAi, input_gate, temp0, seq_len);
  ElementSub(dAi, temp0, dAi, seq_len);

  // dAo = dO * O * (1 - O)
  float *dAo = temp2;
  ElementMul(dO, output_gate, dAo, seq_len);
  ElementMul(dAo, output_gate, temp0, seq_len);
  ElementSub(dAo, temp0, dAo, seq_len);

  // dAf = dF * F * (1 - F)
  float *dAf = temp3;
  ElementMul(dF, forget_gate, dAf, seq_len);
  ElementMul(dAf, forget_gate, temp0, seq_len);
  ElementSub(dAf, temp0, dAf, seq_len);

  float *dAg = temp4;
  ElementMul(cell_gate, cell_gate, temp0, seq_len);
  ElementMul(dG, temp0, temp0, seq_len);
  ElementSub(dG, temp0, dAg, seq_len);

  float *mat_workspace = AllocteFromScrachPad(
    &scratchPad, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_));
  float *weights_loop = w;
  float *dA_loop = dAi;  // dAi, dAo, dAf, dAg
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 0, lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->input_size_, 1.0, dX, lstm_param->input_size_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->input_size_;
    dA_loop += seq_len;
  }

  // calculate dH next
  size_t dH_size = lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float);
  memset(dH, 0, dH_size);
  dA_loop = dAi;
  weights_loop = v;
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->hidden_size_, 1.0, dH, lstm_param->hidden_size_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->hidden_size_;
    dA_loop += seq_len;
  }
  // calculate dC next
  ElementMul(dC, forget_gate, dC, seq_len);
}

void LstmGradDoWeightStep(float *input_t, float *prev_hidden_state, float *dA, float *dW, float *dV, float *dB,
                          float *workspace, const LstmGradParameter *lstm_param) {
  //  Calc dWi, dWo, dWf, dWg, dVi, dVo, dVf, dVg, dBi, dBo, dBf, dBg
  int seq_len = lstm_param->batch_ * lstm_param->hidden_size_;
  float *mat_workspace = AllocteFromScrachPad(
    &workspace, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_));
  float *dA_loop = dA;  // dAi, dAo, dAf, dAg
  int dW_size = lstm_param->input_size_ * lstm_param->hidden_size_;
  int dV_size = lstm_param->hidden_size_ * lstm_param->hidden_size_;
  int dB_size = 0;
  float *dW_loop = dW;
  float *dV_loop = dV;
  float *dB_loop = 0;
  if (lstm_param->has_bias_) {
    dB_loop = dB;
    dB_size = lstm_param->hidden_size_;
  }

  for (int idx = 0; idx < num_of_gates; idx++) {
    // Calc dW
    GemmMatmul(1, 0, lstm_param->hidden_size_, lstm_param->input_size_, lstm_param->batch_, 1.0, dA_loop,
               lstm_param->hidden_size_, input_t, lstm_param->input_size_, 1.0, dW_loop, lstm_param->input_size_,
               mat_workspace);
    // Calc dV
    GemmMatmul(1, 0, lstm_param->hidden_size_, lstm_param->hidden_size_, lstm_param->batch_, 1.0, dA_loop,
               lstm_param->hidden_size_, prev_hidden_state, lstm_param->hidden_size_, 1.0, dV_loop,
               lstm_param->hidden_size_, mat_workspace);
    // Clac dB
    if (dB_loop != 0) {
      sumCols(lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, dA_loop, dB_loop, true);
    }
    dA_loop += seq_len;
    dW_loop += dW_size;
    dV_loop += dV_size;
    dB_loop += dB_size;
  }
}
