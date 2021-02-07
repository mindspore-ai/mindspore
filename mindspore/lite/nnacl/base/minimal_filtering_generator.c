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
#include "nnacl/base/minimal_filtering_generator.h"
#include <string.h>
#include <math.h>
#include "nnacl/fp32/winograd_utils.h"
#include "nnacl/errorcode.h"

void Polynomial(const float *interval, float *m, int degree) {
  for (int i = 0; i < degree; ++i) {
    float mul = 1;
    for (int j = 0; j < degree; ++j) {
      if (i == j) {
        continue;
      }
      mul *= (interval[i] - interval[j]);
    }
    m[i] = mul;
  }
}

void DiagonalPlusMatrix(const float *matrix, float *diagonal_matrix, int degree) {
  int data_num = (degree + 1) * (degree + 1);
  memset(diagonal_matrix, 0, data_num * sizeof(float));
  for (int i = 0; i < degree; ++i) {
    for (int j = 0; j < degree; ++j) {
      if (j == i) {
        diagonal_matrix[i * (degree + 1) + j] = matrix[i];
      }
    }
  }
  diagonal_matrix[data_num - 1] = 1;
}

void ResidueMatrix(const float *interval, float *b, int row, int col) {
  // row : input unit, col : output_unit
  // result : matrix b
  int len = row * col;
  memset(b, 0, len * sizeof(float));
  for (int i = 0; i < row - 1; ++i) {
    for (int j = 0; j < col; ++j) {
      b[i * col + j] = pow(interval[i], j);
    }
  }
  b[len - 1] = 1;
}

int LT(const float *poly_array, float *matrix_lt, int n) {
  if (n > MAX_LEN) {
    return NNACL_ERR;
  }
  float coefficient_array[MAX_LEN];  // n
  float poly[MAX_LEN];               // n

  Polynomial(poly_array, poly, n);
  for (int i = 0; i < n; ++i) {
    // get coefficient
    int index = 1;
    memset(coefficient_array, 0, n * sizeof(float));
    coefficient_array[0] = 1;
    for (int j = 0; j < n; ++j) {
      if (j == i) continue;
      float poly_coe = poly_array[j] == 0 ? 0 : -poly_array[j];
      coefficient_array[index] = 1;
      for (int k = index - 1; k > 0; --k) {
        coefficient_array[k] = coefficient_array[k] * poly_coe + coefficient_array[k - 1];
      }
      coefficient_array[0] *= poly_coe;
      index++;
    }

    // lx[i, 0].nth(j) / f[i]
    int setp = i * n;
    for (int l = 0; l < n; ++l) {
      matrix_lt[setp + l] = coefficient_array[l] / poly[i];
    }
  }  // matrix L row loop
  return NNACL_OK;
}

void T(const float *poly_array, float *matrix_t, int n) {
  memset(matrix_t, 0, n * (n + 1) * sizeof(float));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n + 1; ++j) {
      if (j == i) matrix_t[i * (n + 1) + j] = 1;
      if (j == n) {
        if (poly_array[i] == 0) {
          matrix_t[i * (n + 1) + j] = 0;
        } else {
          matrix_t[i * (n + 1) + j] = -pow(poly_array[i], n);
        }
      }
    }
  }
}

int B(const float *poly_array, float *matrix_b, int in_unit) {
  memset(matrix_b, 0, in_unit * in_unit * sizeof(float));
  int n = in_unit - 1;
  if ((n * n) > MAX_LEN || (n * in_unit) > MAX_LEN) {
    return NNACL_ERR;
  }
  float matrix_l[MAX_LEN];   // n * n
  float matrix_lt[MAX_LEN];  // n * n
  float matrix_t[MAX_LEN];   // n * in_unit

  T(poly_array, matrix_t, n);
  LT(poly_array, matrix_lt, n);
  MatrixTranspose(matrix_lt, matrix_l, n, n);
  MatrixMultiply(matrix_l, matrix_t, matrix_b, n, n, in_unit);
  matrix_b[in_unit * in_unit - 1] = 1;
  return NNACL_OK;
}

#if !defined(ENABLE_ARM) && !defined(ENABLE_SSE)
void MatrixMultiplyWinograd(const float *matix_a, const float *matrix_b, float *matrix_c, int m, int k, int n,
                            int in_channel, int c4_channel) {
  int cnt = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int y = 0; y < in_channel; ++y) {
        float tmp = 0;
        for (int z = 0; z < k; ++z) {
          tmp += matix_a[z * in_channel + y + i * in_channel * k] * matrix_b[j + z * n];
        }
        matrix_c[cnt++] = tmp;
      }
      cnt += c4_channel / 4 - in_channel;
    }
  }
}
#endif

void GenerateIntervalArray(float *array, float interval, int degree) {
  array[0] = 0;
  for (int i = 1; i < degree; ++i) {
    int coefficient = pow(-1, i - 1);
    array[i] = array[i - 1] + interval * i * coefficient;
  }
}

void MatrixTranspose(const float *matrix, float *trans_matrix, int row, int col) {
  for (int i = 0; i < col; ++i) {
    for (int j = 0; j < row; ++j) {
      trans_matrix[i * row + j] = matrix[j * col + i];
    }
  }
}

void MatrixMultiply(const float *matrix_a, const float *matrix_b, float *matrix_c, int m, int k, int n) {
  int count = 0;
  for (int h = 0; h < m; h++) {
    int h_offset = h * k;
    for (int w = 0; w < n; w++) {
      float res = 0;
      for (int i = 0; i < k; i++) {
        res += *(matrix_a + h_offset + i) * *(matrix_b + w + i * n);
      }
      *(matrix_c + count) = res;
      count++;
    }
  }
}

int CookToomFilter(float *matrix_a, float *matrix_at, float *matrix_b, float *matrix_bt, float *matrix_g,
                   float *matrix_gt, float coefficient, int out_unit, int filter_size) {
  int in_unit = out_unit + filter_size - 1;
  int degree = in_unit - 1;
  if (degree > MAX_LEN || (in_unit * in_unit) > MAX_LEN || (in_unit * filter_size) > MAX_LEN) {
    return NNACL_ERR;
  }
  float polynomial_m[MAX_LEN];             // degree
  float diagonal_matrix[MAX_LEN];          // input_unit * input_unit
  float inverse_diagonal_matrix[MAX_LEN];  // input_unit * input_unit

  // get diagonal matrix
  float interval[MAX_LEN];  // degree
  GenerateIntervalArray(interval, coefficient, degree);
  Polynomial(interval, polynomial_m, degree);
  DiagonalPlusMatrix(polynomial_m, diagonal_matrix, degree);
  if (diagonal_matrix[0] < 0) {
    for (int i = 0; i < in_unit; ++i) {
      if (diagonal_matrix[i] != 0) diagonal_matrix[i] *= -1;
    }
  }

  // inverse diagonal matrix
  for (int j = 0; j < in_unit * in_unit; ++j) {
    if (diagonal_matrix[j] != 0) {
      inverse_diagonal_matrix[j] = 1.0 / diagonal_matrix[j];
    } else {
      inverse_diagonal_matrix[j] = 0;
    }
  }

  // get matrix A && AT
  ResidueMatrix(interval, matrix_a, in_unit, out_unit);
  MatrixTranspose(matrix_a, matrix_at, in_unit, out_unit);

  // get matrix B
  int ret = B(interval, matrix_bt, in_unit);
  if (ret != NNACL_OK) {
    return ret;
  }
  MatrixTranspose(matrix_bt, matrix_b, in_unit, in_unit);
  MatrixMultiply(diagonal_matrix, matrix_b, matrix_bt, in_unit, in_unit, in_unit);
  MatrixTranspose(matrix_bt, matrix_b, in_unit, in_unit);

  // get matrix G && GT
  float tmp_g[MAX_LEN];  // in_unit * filter_size
  ResidueMatrix(interval, matrix_g, in_unit, filter_size);
  MatrixTranspose(matrix_g, tmp_g, in_unit, filter_size);
  MatrixMultiply(tmp_g, inverse_diagonal_matrix, matrix_gt, filter_size, in_unit, in_unit);
  MatrixTranspose(matrix_gt, matrix_g, filter_size, in_unit);
  return NNACL_OK;
}

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void MatrixMultiplyVec(const MS_FLOAT32X4 *matrix_a, const MS_FLOAT32X4 *matrix_b, MS_FLOAT32X4 *matrix_c,
                       const float *bias, int m, int k, int n) {
  int count = 0;
  MS_FLOAT32X4 bias_ptr = MS_MOVQ_F32(0);
  if (bias != NULL) {
    bias_ptr = MS_LDQ_F32(bias);
  }
  for (int h = 0; h < m; h++) {
    int h_offset = h * k;
    for (int w = 0; w < n; w++) {
      MS_FLOAT32X4 res = MS_MOVQ_F32(0);
      for (int i = 0; i < k; i++) {
        res = MS_MLAQ_F32(res, matrix_a[h_offset + i], matrix_b[w + i * n]);
      }
      matrix_c[count] = MS_ADDQ_F32(res, bias_ptr);
      count++;
    }
  }
}
#endif

int WinogradWeightTransform(const float *weight_data, float *winograd_data, float *matrix_g, const float *matrix_gt,
                            int oc_block, int input_unit, int kernel_unit, int channel, int batch, bool pack) {
  if (oc_block == 0) {
    return NNACL_PARAM_INVALID;
  }
  // original weight format : ohwi
  int oc_block_num = UP_DIV(batch, oc_block);
  int block_stride = channel * oc_block;
  int block_num_stride = block_stride * oc_block_num;

  // trans_filter = G*g*GT (g represents weight_data)
  // separate into two steps ===> tmp = (g * GT)T ===> trans = (tmp * GT)T   use same function:MatrixMultiplyWinograd
  float *tmp_data = (float *)(malloc(channel * input_unit * kernel_unit * sizeof(float)));
  if (tmp_data == NULL) {
    return NNACL_ERR;
  }
  float *trans_out_data = (float *)(malloc(channel * input_unit * input_unit * sizeof(float)));
  if (trans_out_data == NULL) {
    free(tmp_data);
    return NNACL_ERR;
  }

#ifndef ENABLE_ARM
  float *tmp_data1 = (float *)(malloc(channel * input_unit * kernel_unit * sizeof(float)));
  if (tmp_data1 == NULL) {
    free(tmp_data);
    free(trans_out_data);
    return NNACL_ERR;
  }
  float *trans_out_data1 = (float *)(malloc(channel * input_unit * input_unit * sizeof(float)));
  if (trans_out_data1 == NULL) {
    free(tmp_data);
    free(tmp_data1);
    free(trans_out_data);
    return NNACL_ERR;
  }
#endif

  int input_oz_offset = kernel_unit * kernel_unit * channel;
  for (int i = 0; i < batch; i++) {
    int out_c_block = i / oc_block;
    int out_c_res = i % oc_block;
    int output_oz_offset = out_c_block * block_stride + out_c_res;

#ifndef ENABLE_ARM
    // tmp_data = g * GT
    MatrixMultiplyWinograd(weight_data + i * input_oz_offset, matrix_gt, tmp_data, kernel_unit, kernel_unit, input_unit,
                           channel, channel * 4);
    // tmp_data1 = (tmp_data)T
    PackHWCToWHC(tmp_data, tmp_data1, kernel_unit, input_unit, channel);
    // trans_out_data1 = tmp * GT
    MatrixMultiplyWinograd(tmp_data1, matrix_gt, trans_out_data1, input_unit, kernel_unit, input_unit, channel,
                           channel * 4);
    // trans_out_data = (trans_out_data1)T
    PackHWCToWHC(trans_out_data1, trans_out_data, input_unit, input_unit, channel);
#else
    // tmp = (g * GT)T
    MatrixMultiplyWinograd(weight_data + i * input_oz_offset, matrix_gt, tmp_data, kernel_unit, kernel_unit, input_unit,
                           channel, channel * 4);
    // trans = (tmp * GT)T
    MatrixMultiplyWinograd(tmp_data, matrix_gt, trans_out_data, input_unit, kernel_unit, input_unit, channel,
                           channel * 4);
#endif
    if (pack) {
      int in_offset = 0;
      for (int j = 0; j < input_unit; ++j) {
        for (int k = 0; k < input_unit; ++k) {
          for (int c = 0; c < channel; ++c) {
            *(winograd_data + output_oz_offset + c * oc_block) = trans_out_data[in_offset + c];
          }
          in_offset += channel;
          output_oz_offset += block_num_stride;
        }
      }
    } else {
      memcpy(winograd_data + i * channel * input_unit * input_unit, trans_out_data,
             channel * input_unit * input_unit * sizeof(float));
    }
  }
#ifndef ENABLE_ARM
  free(tmp_data1);
  free(trans_out_data1);
#endif
  free(tmp_data);
  free(trans_out_data);
  return NNACL_OK;
}
