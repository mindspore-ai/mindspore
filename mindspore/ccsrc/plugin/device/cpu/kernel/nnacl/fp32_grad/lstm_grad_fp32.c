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
  temp = MatSizeTotal(hidden_size, batch, hidden_size, batch);
  workspace_size = (temp > workspace_size) ? temp : workspace_size;
  temp = MatSizeTotal(hidden_size, input_size, batch, input_size);
  workspace_size = (temp > workspace_size) ? temp : workspace_size;
  temp = MatSizeTotal(hidden_size, hidden_size, batch, hidden_size);
  workspace_size = (temp > workspace_size) ? temp : workspace_size;
  return workspace_size;
}

int GetRunWorkspaceSize(const LstmGradParameter *lstm_param) {
  int workspace_size = no_of_temp_matrices_sized_output_step * lstm_param->output_step_;
  workspace_size += GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_);
  return workspace_size;
}

size_t GetRunWorkspaceGemmOffset(const LstmGradParameter *lstm_param) {
  return no_of_temp_matrices_sized_output_step * lstm_param->output_step_;
}

void LstmGradDoInputStep(const float *output_gate, float *cell_state, float *prev_cell_state, float *cell_gate,
                         float *input_gate, float *forget_gate, float *dY, float *dC, float *dH, float **dA, float *dX,
                         float *weights, float *workspace, const LstmGradParameter *lstm_param) {
  float *scratchPad = workspace;

  float *temp0 = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  float *temp1 = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  float *temp2 = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  float *temp3 = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  float *temp4 = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);

  // Accumulate gradients into dH
  ElementAdd(dH, dY, dH, lstm_param->output_step_);

  ElementMul(dH, output_gate, temp1, lstm_param->output_step_);
  Tanh(cell_state, lstm_param->output_step_, temp0);
  ElementMul(temp0, temp0, temp2, lstm_param->output_step_);
  ElementMul(temp1, temp2, temp4, lstm_param->output_step_);
  ElementSub(temp1, temp4, temp1, lstm_param->output_step_);
  ElementAdd(dC, temp1, dC, lstm_param->output_step_);

  // calculate dI, dO, dF and dG
  float *dI = temp1;  // dI = dC_{t} * G
  ElementMul(dC, cell_gate, dI, lstm_param->output_step_);
  float *dO = temp2;  // dO = dH * Tanh(C_{t})
  ElementMul(dH, temp0, dO, lstm_param->output_step_);
  float *dF = temp3;  // dF = dC_{t} * C_{t-1}
  ElementMul(dC, prev_cell_state, dF, lstm_param->output_step_);
  float *dG = temp4;  // dG = dC_{t} * I
  ElementMul(dC, input_gate, dG, lstm_param->output_step_);

  // dAi = dI * I * (1 - I)
  float *dAi = temp1;
  *dA = dAi;
  ElementMul(dI, input_gate, dAi, lstm_param->output_step_);
  ElementMul(dAi, input_gate, temp0, lstm_param->output_step_);
  ElementSub(dAi, temp0, dAi, lstm_param->output_step_);

  // dAo = dO * O * (1 - O)
  float *dAo = temp2;
  ElementMul(dO, output_gate, dAo, lstm_param->output_step_);
  ElementMul(dAo, output_gate, temp0, lstm_param->output_step_);
  ElementSub(dAo, temp0, dAo, lstm_param->output_step_);

  // dAf = dF * F * (1 - F)
  float *dAf = temp3;
  ElementMul(dF, forget_gate, dAf, lstm_param->output_step_);
  ElementMul(dAf, forget_gate, temp0, lstm_param->output_step_);
  ElementSub(dAf, temp0, dAf, lstm_param->output_step_);

  float *dAg = temp4;
  ElementMul(cell_gate, cell_gate, temp0, lstm_param->output_step_);
  ElementMul(dG, temp0, temp0, lstm_param->output_step_);
  ElementSub(dG, temp0, dAg, lstm_param->output_step_);

  // calculate dX
  size_t dX_size = lstm_param->batch_ * lstm_param->input_size_ * sizeof(float);
  memset(dX, 0, dX_size);
  float *mat_workspace = AllocteFromScrachPad(
    &scratchPad, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_));
  float *weights_loop = weights;
  float *dA_loop = dAi;  // dAi, dAo, dAf, dAg
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 0, lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->input_size_, 1.0, dX, lstm_param->input_size_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->input_size_;
    dA_loop += lstm_param->output_step_;
  }

  // calculate dH next
  size_t dH_size = lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float);
  memset(dH, 0, dH_size);
  dA_loop = dAi;
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->hidden_size_, 1.0, dH, lstm_param->hidden_size_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->hidden_size_;
    dA_loop += lstm_param->output_step_;
  }
  // calculate dC next
  ElementMul(dC, forget_gate, dC, lstm_param->output_step_);
}

void LstmGradDoWeightStep(float *input_t, float *prev_hidden_state, float *dA, float *dW, float *workspace,
                          const LstmGradParameter *lstm_param) {
  //  Calc dWi, dWo, dWf, dWg, dVi, dVo, dVf, dVg, dBi, dBo, dBf, dBg
  float *mat_workspace = AllocteFromScrachPad(
    &workspace, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_));
  float *dA_loop = dA;  // dAi, dAo, dAf, dAg
  int dW_size = lstm_param->input_size_ * lstm_param->hidden_size_;
  int dV_size = lstm_param->hidden_size_ * lstm_param->hidden_size_;
  int dB_size = 0;
  float *dW_loop = dW;
  float *dV_loop = dW + (num_of_gates * dW_size);
  float *dB_loop = 0;
  if (lstm_param->has_bias_) {
    dB_loop = dW + (num_of_gates * (dW_size + dV_size));
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
    dA_loop += lstm_param->output_step_;
    dW_loop += dW_size;
    dV_loop += dV_size;
    dB_loop += dB_size;
  }
}

void LstmGradDoStep(const float *output_gate, float *cell_state, float *cell_state_minus1, float *cell_gate,
                    float *input_gate, float *forget_gate, float *dY, float *dC, float *dH, float *dX, float *weights,
                    float *dW, float *hidden_state, float *input_t, float *workspace,
                    const LstmGradParameter *lstm_param) {
  float *workspace_i = workspace;

  float buffer[1024];
  float *scratchPad = buffer;

  float *tanh_c = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  float *temp = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  float *temp2 = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  float *tanh_c_sqr = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);

  // Accumulate gradients into dH
  ElementAdd(dH, dY, dH, lstm_param->output_step_);

  ElementMul(dH, output_gate, temp2, lstm_param->output_step_);
  Tanh(cell_state, lstm_param->output_step_, tanh_c);
  ElementMul(tanh_c, tanh_c, tanh_c_sqr, lstm_param->output_step_);
  ElementMul(temp2, tanh_c_sqr, temp, lstm_param->output_step_);
  ElementSub(temp2, temp, temp2, lstm_param->output_step_);
  ElementAdd(dC, temp2, dC, lstm_param->output_step_);

  // calculate dI, dO, dF and dG
  float *dI = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);  // dI = dC_{t} * G
  ElementMul(dC, cell_gate, dI, lstm_param->output_step_);
  float *dO = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);  // dO = dH * Tanh(C_{t})
  ElementMul(dH, tanh_c, dO, lstm_param->output_step_);
  float *dF = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);  // dF = dC_{t} * C_{t-1}
  ElementMul(dC, cell_state_minus1, dF, lstm_param->output_step_);
  float *dG = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);  // dG = dC_{t} * I
  ElementMul(dC, input_gate, dG, lstm_param->output_step_);

  // dAi = dI * I * (1 - I)
  float *dAi = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  ElementMul(dI, input_gate, dAi, lstm_param->output_step_);
  ElementMul(dAi, input_gate, temp, lstm_param->output_step_);
  ElementSub(dAi, temp, dAi, lstm_param->output_step_);

  // dAo = dO * O * (1 - O)
  float *dAo = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  ElementMul(dO, output_gate, dAo, lstm_param->output_step_);
  ElementMul(dAo, output_gate, temp, lstm_param->output_step_);
  ElementSub(dAo, temp, dAo, lstm_param->output_step_);

  // dAf = dF * F * (1 - F)
  float *dAf = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  ElementMul(dF, forget_gate, dAf, lstm_param->output_step_);
  ElementMul(dAf, forget_gate, temp, lstm_param->output_step_);
  ElementSub(dAf, temp, dAf, lstm_param->output_step_);

  // dAg = dG * (1 - G^2)
  float *dAg = AllocteFromScrachPad(&scratchPad, lstm_param->output_step_);
  ElementMul(cell_gate, cell_gate, dAg, lstm_param->output_step_);
  ElementMul(dG, dAg, dAg, lstm_param->output_step_);
  ElementSub(dG, dAg, dAg, lstm_param->output_step_);

  // calculate dX
  float *mat_workspace = AllocteFromScrachPad(
    &workspace_i, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_));
  float *weights_loop = weights;
  float *dA_loop = dAi;  // dAi, dAo, dAf, dAg
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 0, lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->input_size_, 1.0, dX, lstm_param->input_size_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->input_size_;
    dA_loop += lstm_param->output_step_;
  }

  // calculate dH next
  size_t dH_size = lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float);
  memset(dH, 0, dH_size);
  dA_loop = dAi;
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->hidden_size_, 1.0, dH, lstm_param->hidden_size_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->hidden_size_;
    dA_loop += lstm_param->output_step_;
  }
  // calculate dC next
  ElementMul(dC, forget_gate, dC, lstm_param->output_step_);

  //  Calc dWi, dWo, dWf, dWg, dVi, dVo, dVf, dVg, dBi, dBo, dBf, dBg
  dA_loop = dAi;
  int dW_size = lstm_param->input_size_ * lstm_param->hidden_size_;
  int dV_size = lstm_param->hidden_size_ * lstm_param->hidden_size_;
  int dB_size = lstm_param->hidden_size_;
  float *dW_loop = dW;
  float *dV_loop = dW + (num_of_gates * dW_size);
  float *dB_loop = dW + (num_of_gates * (dW_size + dV_size));
  for (int idx = 0; idx < num_of_gates; idx++) {
    // Calc dW
    GemmMatmul(1, 0, lstm_param->hidden_size_, lstm_param->input_size_, lstm_param->batch_, 1.0, dA_loop,
               lstm_param->hidden_size_, input_t, lstm_param->input_size_, 1.0, dW_loop, lstm_param->input_size_,
               mat_workspace);
    // Calc dV
    if (hidden_state != 0) {
      GemmMatmul(1, 0, lstm_param->hidden_size_, lstm_param->hidden_size_, lstm_param->batch_, 1.0, dA_loop,
                 lstm_param->hidden_size_, hidden_state, lstm_param->hidden_size_, 1.0, dV_loop,
                 lstm_param->hidden_size_, mat_workspace);
    }
    // Clac dB
    sumCols(lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, dA_loop, dB_loop, true);
    dA_loop += lstm_param->output_step_;
    dW_loop += dW_size;
    dV_loop += dV_size;
    dB_loop += dB_size;
  }
}
