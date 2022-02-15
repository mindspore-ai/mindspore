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

static const int no_of_temp_matrices_sized_output_step = 10;
static const int num_of_gates = 4;

static inline float *AllocteFromScrachPad(float **scrach_pad, int size) {
  float *buffer = *scrach_pad;
  *scrach_pad += size;
  return buffer;
}

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

int GetRunWorkspaceSize(const LstmParameter *lstm_param) {
  int workspace_size = no_of_temp_matrices_sized_output_step * lstm_param->output_step_;
  workspace_size += GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_);
  return workspace_size;
}

void LstmGradStepUnitInit(const float *output_gate, float *cell_state, float *dY, float *dC, float *dH,
                          float *workspace, const LstmParameter *lstm_param) {
  int state_size = lstm_param->batch_ * lstm_param->hidden_size_;
  memcpy(dH, dY, state_size * sizeof(float));
  float *workspace_i = workspace;
  float *tanh_c = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);
  float *temp = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);
  ElementMul(dH, output_gate, dC, lstm_param->output_step_);
  Tanh(cell_state, lstm_param->output_step_, tanh_c);
  ElementMul(tanh_c, tanh_c, tanh_c, lstm_param->output_step_);
  ElementMul(dC, tanh_c, temp, lstm_param->output_step_);
  ElementSub(dC, temp, dC, lstm_param->output_step_);
}

void LstmGradStepUnit(float *output, float *input_gate, float *forget_gate, float *cell_gate, float *output_gate,
                      float *hidden_state, float *cell_state, float *dC, float *dH, float *dY, float *dX,
                      float *cell_state_minus1, float *weights, float *workspace, float *dA,
                      const LstmParameter *lstm_param) {
  float *workspace_i = workspace;

  float *dI = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dI = dC_{t+1} * G
  ElementMul(dC, cell_gate, dI, lstm_param->output_step_);
  float *tanh_c = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);
  Tanh(cell_state, lstm_param->output_step_, tanh_c);
  float *dO = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dO = dH * Tanh(C_{t+1})
  ElementMul(dH, tanh_c, dO, lstm_param->output_step_);
  float *dF = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dF = dC_{t+1} * C_t
  ElementMul(dC, cell_state_minus1, dF, lstm_param->output_step_);
  float *dG = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dG = dC_{t+1} * I
  ElementMul(dC, input_gate, dG, lstm_param->output_step_);

  float *temp = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);
  float *dAi = AllocteFromScrachPad(&dA, lstm_param->output_step_);  // dAi = dI * I * (1 - I)
  ElementMul(dI, input_gate, dAi, lstm_param->output_step_);
  ElementMul(dAi, input_gate, temp, lstm_param->output_step_);
  ElementSub(dAi, temp, dAi, lstm_param->output_step_);
  float *dAo = AllocteFromScrachPad(&dA, lstm_param->output_step_);  // dAo = dO * O * (1 - O)
  ElementMul(dO, output_gate, dAo, lstm_param->output_step_);
  ElementMul(dAo, output_gate, temp, lstm_param->output_step_);
  ElementSub(dAo, temp, dAo, lstm_param->output_step_);
  float *dAf = AllocteFromScrachPad(&dA, lstm_param->output_step_);  // dAf = dF * F * (1 - F)
  ElementMul(dF, forget_gate, dAf, lstm_param->output_step_);
  ElementMul(dAf, forget_gate, temp, lstm_param->output_step_);
  ElementSub(dAf, temp, dAf, lstm_param->output_step_);
  float *dAg = AllocteFromScrachPad(&dA, lstm_param->output_step_);  // dAg = dG * (1 - G^2)
  ElementMul(cell_gate, cell_gate, dAg, lstm_param->output_step_);
  ElementMul(dG, dAg, dAg, lstm_param->output_step_);
  ElementSub(dG, dAg, dAg, lstm_param->output_step_);

  float *mat_workspace = AllocteFromScrachPad(
    &workspace_i, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_));

  size_t dX_size = lstm_param->batch_ * lstm_param->input_size_ * sizeof(float);
  memset(dX, 0, dX_size);

  float *weights_loop = weights;
  float *dA_loop = dAi;  // dAi, dAo, dAf, dAg
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 0, lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->input_size_, 1.0, dX, lstm_param->input_size_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->input_size_;
    dA_loop += lstm_param->output_step_;
  }

  size_t dH_size = lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float);
  if (dY != NULL) {
    memcpy(dH, dY, dH_size);
    output_gate -= lstm_param->output_step_;
  } else {
    memset(dH, 0, dH_size);
  }
  dA_loop = dAi;
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 1, lstm_param->hidden_size_, lstm_param->batch_, lstm_param->hidden_size_, 1.0, weights_loop,
               lstm_param->hidden_size_, dA_loop, lstm_param->hidden_size_, 1.0, dH, lstm_param->batch_, mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->hidden_size_;
    dA_loop += lstm_param->output_step_;
  }

  NNACL_ASSERT(workspace_i <= workspace + GetRunWorkspaceSize(lstm_param));

  ElementMul(dC, forget_gate, dC, lstm_param->output_step_);
  ElementMul(dH, output_gate, temp, lstm_param->output_step_);

  Tanh(cell_state_minus1, lstm_param->output_step_, tanh_c);
  ElementMul(tanh_c, tanh_c, tanh_c, lstm_param->output_step_);
  ElementMul(temp, tanh_c, tanh_c, lstm_param->output_step_);
  ElementSub(temp, tanh_c, temp, lstm_param->output_step_);
  ElementAdd(dC, temp, dC, lstm_param->output_step_);
}

void LstmGradWeightStepUnit(float *input_t, float *hidden_state, float *dA, float *dW, float *workspace,
                            const LstmParameter *lstm_param) {
  // Calc dWi, dWo, dWf, dWg, dVi, dVo, dVf, dVg, dBi, dBo, dBf, dBg
  float *dA_loop = dA;
  float *mat_workspace = AllocteFromScrachPad(
    &workspace, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->input_size_, lstm_param->hidden_size_));
  int dW_size = lstm_param->input_size_ * lstm_param->hidden_size_;
  int dV_size = lstm_param->hidden_size_ * lstm_param->hidden_size_;
  int dB_size = lstm_param->hidden_size_;
  float *dW_loop = dW;
  float *dV_loop = dW + (num_of_gates * dW_size);
  float *dB_loop = dW + (num_of_gates * (dW_size + dV_size));
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(1, 0, lstm_param->hidden_size_, lstm_param->input_size_, lstm_param->batch_, 1.0, dA_loop,
               lstm_param->hidden_size_, input_t, lstm_param->input_size_, 1.0, dW_loop, lstm_param->input_size_,
               mat_workspace);  // Calc dW
    GemmMatmul(1, 0, lstm_param->hidden_size_, lstm_param->hidden_size_, lstm_param->batch_, 1.0, dA_loop,
               lstm_param->hidden_size_, hidden_state, lstm_param->hidden_size_, 1.0, dV_loop, lstm_param->hidden_size_,
               mat_workspace);                                                                                // Calc dV
    sumCols(lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, dA_loop, dB_loop, true);  // Clac dB
    dA_loop += lstm_param->output_step_;
    dW_loop += dW_size;
    dV_loop += dV_size;
    dB_loop += dB_size;
  }
}
