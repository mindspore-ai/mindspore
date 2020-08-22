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

#include "nnacl/fp32/strassen_matmul.h"

bool CheckRecursion(int row, int col, int deep, int max_recursion, int cur_recursion) {
  if (cur_recursion >= max_recursion) {
    return false;
  }

  if (row % 2 != 0 || col % 2 != 0 || deep % 2 != 0) {
    return false;
  }

  int row2 = row / 2;
  int col2 = col / 2;
  int deep2 = deep / 2;

  float save_cost = row * col * 4 * deep * 4 * 2 + row * col * 4 -
                    7 * (row2 * col2 * 4 * deep2 * 4 * 2 - row2 * col2 * 4) - 4 * (row2 * deep2 * 4 * 3) -
                    4 * (deep2 * 4 * col2 * 4 * 3) - 7 * (row2 * col2 * 4 * 3);

  return (save_cost > 0.f);
}

void GemmMatMulComm(const float *a_ptr, const float *b_ptr, float *dst_ptr, int row, int col, int deep, int b_stride,
                    int c_stride) {
  int row4mod = row % 4;
  int row4div = row / 4;
  for (int r = 0; r < row; r++) {
    int r4mod = r % 4;
    int r4div = r / 4;
    for (int c = 0; c < col * 4; c++) {
      float value = 0;
      int ic = c / 4 * c_stride + r * 4 + c % 4;
      for (int d = 0; d < deep * 4; d++) {
        int d4mod = d % 4;
        int d4div = d / 4;
        int a_stride = (r < (row4div * 4)) ? 4 : row4mod;
        int ai = r4div * 4 * deep * 4 + d4div * a_stride * 4 + r4mod * 4 + d4mod;
        int bi = c / 4 * b_stride + d * 4 + c % 4;
        value = value + a_ptr[ai] * b_ptr[bi];
      }
      dst_ptr[ic] = value;
    }
  }
  return;
}

void GemmMatMul(const float *a_ptr, const float *b_ptr, float *dst_ptr, int row, int col, int deep, int b_stride,
                int c_stride) {
  int row4mod = row % 4;
  int row4div = row / 4;

  if (row4div > 0) {
    GemmMatMulComm(a_ptr, b_ptr, dst_ptr, row4div * 4, col, deep, b_stride, c_stride);
  }

  if (row4mod != 0) {
    GemmMatMulComm(a_ptr + row4div * deep * 4 * 4, b_ptr, dst_ptr + row4div * 4 * 4, row4mod, col, deep, b_stride,
                   c_stride);
  }
  return;
}

int RecursionMatmul(const float *a_ptr, const float *b_ptr, float *c_ptr, StrassenMatMulParameter *matmul_param,
                    int max_recursion, int cur_recursion, float *tmp_a_ptr) {
  size_t row2 = matmul_param->row_ / 2;
  size_t deep2 = matmul_param->deep_ / 2;
  size_t col2 = matmul_param->col_ / 2;
  size_t a_stride = matmul_param->a_stride_;
  size_t b_stride = matmul_param->b_stride_;
  size_t c_stride = matmul_param->c_stride_;

  StrassenMatMulParameter rec_matmul;
  rec_matmul.row_ = row2;
  rec_matmul.deep_ = deep2;
  rec_matmul.col_ = col2;

  float *x_ptr = (float *)(malloc(row2 * MSMAX(deep2, col2) * FP32_STRASSEN_UINT * sizeof(float)));
  if (x_ptr == NULL) {
    return NNACL_ERRCODE_STRASSEN_RECURSION_MALLOC;
  }
  float *y_ptr = (float *)(malloc(col2 * deep2 * FP32_STRASSEN_WEIGHT_UINT * sizeof(float)));
  if (y_ptr == NULL) {
    free(x_ptr);
    return NNACL_ERRCODE_STRASSEN_RECURSION_MALLOC;
  }
  size_t x_stride = row2 * FP32_STRASSEN_UINT;
  size_t y_stride = deep2 * FP32_STRASSEN_WEIGHT_UINT;

  const float *a11 = a_ptr;
  const float *a12 = a_ptr + deep2 * a_stride;
  const float *a21 = a_ptr + row2 * FP32_STRASSEN_UINT;
  const float *a22 = a_ptr + deep2 * a_stride + row2 * FP32_STRASSEN_UINT;
  const float *b11 = b_ptr;
  const float *b12 = b_ptr + col2 * b_stride;
  const float *b21 = b_ptr + deep2 * FP32_STRASSEN_WEIGHT_UINT;
  const float *b22 = b_ptr + col2 * b_stride + deep2 * FP32_STRASSEN_WEIGHT_UINT;
  float *c11 = c_ptr;
  float *c12 = c_ptr + col2 * c_stride;
  float *c21 = c_ptr + row2 * FP32_STRASSEN_UINT;
  float *c22 = c_ptr + col2 * c_stride + row2 * FP32_STRASSEN_UINT;

  /* S3 = A11 - A21 */
  MatrixSub(a11, a21, x_ptr, a_stride, a_stride, x_stride, row2, deep2);

  /* T3 = B22 - B12 */
  MatrixSub(b22, b12, y_ptr, b_stride, b_stride, y_stride, deep2 * 4, col2);

  /* P7 = S3T3 */
  rec_matmul.a_stride_ = x_stride;
  rec_matmul.b_stride_ = y_stride;
  rec_matmul.c_stride_ = c_stride;
  StrassenMatmul(x_ptr, y_ptr, c21, &rec_matmul, max_recursion, cur_recursion + 1, tmp_a_ptr);

  /* S1 = A21 + A22 */
  MatrixAdd(a21, a22, x_ptr, a_stride, a_stride, x_stride, row2, deep2);

  /* T1 = B12 - B11 */
  MatrixSub(b12, b11, y_ptr, b_stride, b_stride, y_stride, deep2 * 4, col2);

  /* P5 = S1T1 */
  StrassenMatmul(x_ptr, y_ptr, c22, &rec_matmul, max_recursion, cur_recursion + 1, tmp_a_ptr);

  /* S2 = S1 - A11 */
  MatrixSub(x_ptr, a11, x_ptr, x_stride, a_stride, x_stride, row2, deep2);

  /* T2 = B22 - T1 */
  MatrixSub(b22, y_ptr, y_ptr, b_stride, y_stride, y_stride, deep2 * 4, col2);

  /* P6 = S2T2 */
  StrassenMatmul(x_ptr, y_ptr, c12, &rec_matmul, max_recursion, cur_recursion + 1, tmp_a_ptr);

  /* S4 = A12 - S2 */
  MatrixSub(a12, x_ptr, x_ptr, a_stride, x_stride, x_stride, row2, deep2);

  /* P3 = S4B22 */
  rec_matmul.b_stride_ = b_stride;
  StrassenMatmul(x_ptr, b22, c11, &rec_matmul, max_recursion, cur_recursion + 1, tmp_a_ptr);

  /* P1 = A11B11 */
  rec_matmul.a_stride_ = a_stride;
  rec_matmul.c_stride_ = row2 * FP32_STRASSEN_UINT;
  StrassenMatmul(a11, b11, x_ptr, &rec_matmul, max_recursion, cur_recursion + 1, tmp_a_ptr);

  /* U2 = P1 + P6
     U3 = U2 + P7
     U4 = U2 + P5
     U7 = U3 + P5
     U5 = U4 + P3 */
  MatrixMultiAdd(c11, c12, c21, c22, x_ptr, row2, col2, c_stride, x_stride);

  /* T4 = T2 - B21 */
  MatrixSub(y_ptr, b21, y_ptr, y_stride, b_stride, y_stride, deep2 * 4, col2);

  /* P4 = A22T4 */
  rec_matmul.b_stride_ = y_stride;
  rec_matmul.c_stride_ = c_stride;
  StrassenMatmul(a22, y_ptr, c11, &rec_matmul, max_recursion, cur_recursion + 1, tmp_a_ptr);

  /* U6 = U3 - P4 */
  MatrixSub(c21, c11, c21, c_stride, c_stride, c_stride, row2, col2);

  /* P2 = A12B21 */
  rec_matmul.b_stride_ = b_stride;
  StrassenMatmul(a12, b21, c11, &rec_matmul, max_recursion, cur_recursion + 1, tmp_a_ptr);

  /* U1 = P1 + P2 */
  MatrixAdd(x_ptr, c11, c11, x_stride, c_stride, c_stride, row2, col2);

  free(x_ptr);
  free(y_ptr);
  return NNACL_OK;
}

int CommonMatMul(const float *a_ptr, const float *b_ptr, float *c_ptr, StrassenMatMulParameter *matmul_param,
                 float *tmp_a_ptr) {
  MatrixPack(a_ptr, tmp_a_ptr, matmul_param->row_, matmul_param->deep_, matmul_param->a_stride_);
  GemmMatMul(tmp_a_ptr, b_ptr, c_ptr, matmul_param->row_, matmul_param->col_, matmul_param->deep_,
             matmul_param->b_stride_, matmul_param->c_stride_);
  return NNACL_OK;
}

int StrassenMatmul(const float *a_ptr, const float *b_ptr, float *c_ptr, StrassenMatMulParameter *matmul_param,
                   int max_recursion, int cur_recursion, float *tmp_a_ptr) {
  if (CheckRecursion(matmul_param->row_, matmul_param->col_, matmul_param->deep_, cur_recursion, max_recursion)) {
    return RecursionMatmul(a_ptr, b_ptr, c_ptr, matmul_param, max_recursion, cur_recursion, tmp_a_ptr);
  }
  return CommonMatMul(a_ptr, b_ptr, c_ptr, matmul_param, tmp_a_ptr);
}
