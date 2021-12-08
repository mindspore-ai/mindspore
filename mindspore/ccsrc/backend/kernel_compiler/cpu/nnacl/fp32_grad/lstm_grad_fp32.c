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

static const int no_of_temp_matrices_sized_output_step = 15;
static const int no_of_temp_matrices_sized_batch_times_seq_len = 8;  // 4 dW_ and 4 dV_ matrices
static const int num_of_gates = 4;

static inline float *AllocteFromScrachPad(float **scrach_pad, int size) {
  float *buffer = *scrach_pad;
  *scrach_pad += size;
  return buffer;
}

void PackLstmWeightTranspose(float *dst, const float *src, int batch, int col, int row, int row_align) {
  for (int i = 0; i < batch; i++) {
    const float *src_batch = src + i * col * row;
    float *dst_batch = dst + i * col * row_align;
#ifdef ENABLE_AVX
    RowMajor2Row16Major(src_batch, dst_batch, row, col);
#elif defined(ENABLE_ARM32)
    RowMajor2Row4Major(src_batch, dst_batch, row, col);
#else
    RowMajor2Row8Major(src_batch, dst_batch, row, col);
#endif
  }
}

void sumRows(int m, int n, int stride, float *inMat, float *outMat) {
  for (int idm = 0; idm < m; idm++) {
    float *row = inMat + idm * stride;
    *outMat = 0;
    for (int idn = 0; idn < n; idn++) {
      *outMat += *row++;
    }
    outMat++;
  }
}

int GetGemmMatMullWorkspace(int batch, int seq_len, int hidden_size) {
  int workspace_size = MatSizeTotal(batch, seq_len, hidden_size, 0);
  int temp = MatSizeTotal(batch, hidden_size, seq_len, 0);
  workspace_size = (temp > workspace_size) ? temp : workspace_size;
  return workspace_size;
}

int GetRunWorkspaceSize(const LstmParameter *lstm_param) {
  int workspace_size = GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->seq_len_, lstm_param->hidden_size_);
  workspace_size += no_of_temp_matrices_sized_output_step * lstm_param->output_step_;
  workspace_size += no_of_temp_matrices_sized_batch_times_seq_len * lstm_param->batch_ * lstm_param->seq_len_;
  return workspace_size;
}

void LstmGradStepUnit(float *packed_input, float *output, float *input_gate, float *forget_gate, float *cell_gate,
                      float *output_gate, float *hidden_state, float *cell_state, float *dC, float *dH, float *dY,
                      float *cell_state_minus1, float *weights, float *workspace, const LstmParameter *lstm_param) {
  float *workspace_i = workspace;
  float *mat_workspace = AllocteFromScrachPad(
    &workspace_i, GetGemmMatMullWorkspace(lstm_param->batch_, lstm_param->seq_len_, lstm_param->hidden_size_));
  float *tanh_c = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);
  Tanh(cell_state, lstm_param->output_step_, tanh_c);
  float *dO = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dO = dH * Tanh(C_{t+1})
  ElementMul(dH, tanh_c, dO, lstm_param->output_step_);
  float *dF = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dF = dC_{t+1} * C_t
  ElementMul(dC, cell_state_minus1, dF, lstm_param->output_step_);
  float *dG = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dG = dC_{t+1} * I
  ElementMul(dC, input_gate, dG, lstm_param->output_step_);
  float *dI = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dI = dC_{t+1} * G
  ElementMul(dC, cell_gate, dI, lstm_param->output_step_);
  float *dAg = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dAg = dG * (1 - G^2)
  ElementMul(cell_gate, cell_gate, dAg, lstm_param->output_step_);
  ElementMul(dG, dAg, dAg, lstm_param->output_step_);
  ElementSub(dG, dAg, dAg, lstm_param->output_step_);
  float *dAi = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dAi = dI * I * (1 - I)
  float *temp = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);
  ElementMul(dI, input_gate, dAi, lstm_param->output_step_);
  ElementMul(dAi, input_gate, temp, lstm_param->output_step_);
  ElementSub(dAi, temp, dAi, lstm_param->output_step_);
  float *dAf = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dAf = dF * F * (1 - F)
  ElementMul(dF, forget_gate, dAf, lstm_param->output_step_);
  ElementMul(dAf, forget_gate, temp, lstm_param->output_step_);
  ElementSub(dAf, temp, dAf, lstm_param->output_step_);
  float *dAo = AllocteFromScrachPad(&workspace_i, lstm_param->output_step_);  // dAo = dO * O * (1 - O)
  ElementMul(dO, output_gate, dAo, lstm_param->output_step_);
  ElementMul(dAo, output_gate, temp, lstm_param->output_step_);
  ElementSub(dAo, temp, dAo, lstm_param->output_step_);

  float *dX = dY;
  memset(dX, 0, lstm_param->batch_ * lstm_param->seq_len_ * sizeof(float));
  float *weights_loop = weights;
  float *dA_loop = dAg;  // dAg, dAi, dAf, DAo
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 1, lstm_param->batch_, lstm_param->seq_len_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->hidden_size_, 1.0, dX, lstm_param->seq_len_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->seq_len_;
    dA_loop += lstm_param->output_step_;
  }
  float *dWg = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, packed_input,
             lstm_param->batch_, dAg, lstm_param->hidden_size_, 0.0, dWg, lstm_param->hidden_size_, mat_workspace);

  float *dWi = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, packed_input,
             lstm_param->batch_, dAi, lstm_param->hidden_size_, 0.0, dWi, lstm_param->hidden_size_, mat_workspace);

  float *dWf = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, packed_input,
             lstm_param->batch_, dAf, lstm_param->hidden_size_, 0.0, dWf, lstm_param->hidden_size_, mat_workspace);

  float *dWo = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, packed_input,
             lstm_param->batch_, dAo, lstm_param->hidden_size_, 0.0, dWo, lstm_param->hidden_size_, mat_workspace);

  memset(dH, 0, lstm_param->batch_ * lstm_param->hidden_size_ * sizeof(float));
  dA_loop = dAg;
  for (int idx = 0; idx < num_of_gates; idx++) {
    GemmMatmul(0, 1, lstm_param->batch_, lstm_param->seq_len_, lstm_param->hidden_size_, 1.0, dA_loop,
               lstm_param->hidden_size_, weights_loop, lstm_param->hidden_size_, 1.0, dH, lstm_param->seq_len_,
               mat_workspace);
    weights_loop += lstm_param->hidden_size_ * lstm_param->hidden_size_;
    dA_loop += lstm_param->output_step_;
  }
  float *dVg = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, hidden_state,
             lstm_param->batch_, dAg, lstm_param->hidden_size_, 0.0, dVg, lstm_param->hidden_size_, mat_workspace);

  float *dVi = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, hidden_state,
             lstm_param->batch_, dAi, lstm_param->hidden_size_, 0.0, dVi, lstm_param->hidden_size_, mat_workspace);

  float *dVf = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, hidden_state,
             lstm_param->batch_, dAf, lstm_param->hidden_size_, 0.0, dVf, lstm_param->hidden_size_, mat_workspace);

  float *dVo = AllocteFromScrachPad(&workspace_i, lstm_param->batch_ * lstm_param->hidden_size_);
  GemmMatmul(1, 0, lstm_param->batch_, lstm_param->hidden_size_, lstm_param->seq_len_, 1.0, hidden_state,
             lstm_param->batch_, dAo, lstm_param->hidden_size_, 0.0, dVo, lstm_param->hidden_size_, mat_workspace);

  float *dBg = AllocteFromScrachPad(&workspace_i, lstm_param->batch_);
  sumRows(lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, dAg, dBg);
  float *dBi = AllocteFromScrachPad(&workspace_i, lstm_param->batch_);
  sumRows(lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, dAi, dBi);
  float *dBf = AllocteFromScrachPad(&workspace_i, lstm_param->batch_);
  sumRows(lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, dAf, dBf);
  float *dBo = AllocteFromScrachPad(&workspace_i, lstm_param->batch_);
  NNACL_ASSERT(workspace_i <= workspace + GetRunWorkspaceSize(lstm_param));
  sumRows(lstm_param->batch_, lstm_param->hidden_size_, lstm_param->hidden_size_, dAo, dBo);

  ElementMul(dC, forget_gate, dC, lstm_param->output_step_);
  ElementMul(dH, output_gate, temp, lstm_param->output_step_);
  ElementAdd(dC, temp, dC, lstm_param->output_step_);

  Tanh(cell_state_minus1, lstm_param->output_step_, tanh_c);
  ElementMul(tanh_c, tanh_c, tanh_c, lstm_param->output_step_);
  ElementMul(temp, tanh_c, temp, lstm_param->output_step_);
  ElementSub(dC, temp, dC, lstm_param->output_step_);
}
