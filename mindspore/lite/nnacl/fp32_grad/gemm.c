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

#include "nnacl/fp32_grad/gemm.h"
#include <string.h>

static void gemm_not_trana_not_tranb(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b, int ldb,
                                     float *mat_c, int ldc) {
  const int block_size = 4;
  int block_mod = N % block_size;
  int block_c4 = N - block_mod;

  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float a = alpha * mat_a[i * lda + k];
      for (j = 0; j < block_c4; j += block_size) {
        float *b = &mat_b[k * ldb + j];
        float *c = &mat_c[i * ldc + j];
        c[0] += a * b[0];
        c[1] += a * b[1];
        c[2] += a * b[2];
        c[3] += a * b[3];
      }
      for (; j < N; ++j) {
        mat_c[i * ldc + j] += a * mat_b[k * ldb + j];
      }
    }
  }
}

static void gemm_not_trana_tranb(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b, int ldb,
                                 float *mat_c, int ldc) {
  const int block_size = 4;
  int block_mod = K % block_size;
  int block_c4 = K - block_mod;

  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float sum = 0;
      for (k = 0; k < block_c4; k += block_size) {
        float *a = &mat_a[i * lda + k];
        float *b = &mat_b[j * ldb + k];
        sum += alpha * a[0] * b[0];
        sum += alpha * a[1] * b[1];
        sum += alpha * a[2] * b[2];
        sum += alpha * a[3] * b[3];
      }
      for (; k < K; ++k) {
        sum += alpha * mat_a[i * lda + k] * mat_b[j * ldb + k];
      }
      mat_c[i * ldc + j] += sum;
    }
  }
}

static void gemm_trana_not_tranb(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b, int ldb,
                                 float *mat_c, int ldc) {
  const int block_size = 4;
  int block_mod = N % block_size;
  int block_c4 = N - block_mod;

  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float a = alpha * mat_a[k * lda + i];
      for (j = 0; j < block_c4; j += block_size) {
        float *b = &mat_b[k * ldb + j];
        float *c = &mat_c[i * ldc + j];
        c[0] += a * b[0];
        c[1] += a * b[1];
        c[2] += a * b[2];
        c[3] += a * b[3];
      }
      for (; j < N; ++j) {
        mat_c[i * ldc + j] += a * mat_b[k * ldb + j];
      }
    }
  }
}

static void gemm_trana_tranb(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b, int ldb,
                             float *mat_c, int ldc) {
  int i, j, k;
  const int block_size = 4;
  int k_block_mod = K % block_size;
  int k_block_c4 = K - k_block_mod;

  int m_block_mod = M % block_size;
  int m_block_c4 = M - m_block_mod;

  for (i = 0; i < m_block_c4; i += block_size) {
    for (j = 0; j < N; ++j) {
      float sum0 = 0;
      float sum1 = 0;
      float sum2 = 0;
      float sum3 = 0;

      for (k = 0; k < k_block_c4; k += block_size) {
        float *b = &mat_b[j * ldb + k];
        sum0 += alpha * mat_a[i + k * lda] * b[0];
        sum0 += alpha * mat_a[i + (k + 1) * lda] * b[1];
        sum0 += alpha * mat_a[i + (k + 2) * lda] * b[2];
        sum0 += alpha * mat_a[i + (k + 3) * lda] * b[3];

        sum1 += alpha * mat_a[i + 1 + k * lda] * b[0];
        sum1 += alpha * mat_a[i + 1 + (k + 1) * lda] * b[1];
        sum1 += alpha * mat_a[i + 1 + (k + 2) * lda] * b[2];
        sum1 += alpha * mat_a[i + 1 + (k + 3) * lda] * b[3];

        sum2 += alpha * mat_a[i + 2 + k * lda] * b[0];
        sum2 += alpha * mat_a[i + 2 + (k + 1) * lda] * b[1];
        sum2 += alpha * mat_a[i + 2 + (k + 2) * lda] * b[2];
        sum2 += alpha * mat_a[i + 2 + (k + 3) * lda] * b[3];

        sum3 += alpha * mat_a[i + 3 + k * lda] * b[0];
        sum3 += alpha * mat_a[i + 3 + (k + 1) * lda] * b[1];
        sum3 += alpha * mat_a[i + 3 + (k + 2) * lda] * b[2];
        sum3 += alpha * mat_a[i + 3 + (k + 3) * lda] * b[3];
      }
      for (; k < K; ++k) {
        float *b = &mat_b[j * ldb + k];
        sum0 += alpha * mat_a[i + (k * lda)] * b[0];
        sum1 += alpha * mat_a[i + 1 + (k * lda)] * b[0];
        sum2 += alpha * mat_a[i + 2 + (k * lda)] * b[0];
        sum3 += alpha * mat_a[i + 3 + (k * lda)] * b[0];
      }
      mat_c[i * ldc + j] += sum0;
      mat_c[(i + 1) * ldc + j] += sum1;
      mat_c[(i + 2) * ldc + j] += sum2;
      mat_c[(i + 3) * ldc + j] += sum3;
    }
  }
  // no more block of 4x4
  for (; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float sum = 0;
      for (k = 0; k < K; ++k) {
        sum += alpha * mat_a[i + k * lda] * mat_b[k + j * ldb];
      }
      mat_c[i * ldc + j] += sum;
    }
  }
}

// mat_c = alpha*op( mat_a )*op( mat_b ) + beta*C
// M - number of rows of matrix a
// N - number of cols of matrix b
// K - number of cols of matrix a
// lda - fast dim of matrix a
// ldb - fast dim of matrix b
// ldc - fast dim of matrix c
void gemm(int transpose_a, int transpose_b, int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b,
          int ldb, float beta, float *mat_c, int ldc) {
  if (beta >= 0.f && beta <= 0.f) {
    memset(mat_c, 0, M * N * sizeof(float));
  } else if (beta < 1.f || beta > 1.f) {
    const int block_size = 4;
    const int size = M * N;
    int block_mod = size % block_size;
    int block_c4 = size - block_mod;
    int i;
    for (i = 0; i < block_c4; i += block_size) {
      float *c = &mat_c[i];
      c[0] *= beta;
      c[1] *= beta;
      c[2] *= beta;
      c[3] *= beta;
    }
    for (; i < size; ++i) {
      mat_c[i] *= beta;
    }
  }
  if (transpose_a && transpose_b) {
    gemm_trana_tranb(M, N, K, alpha, mat_a, lda, mat_b, ldb, mat_c, ldc);
  } else if (!transpose_a && !transpose_b) {
    gemm_not_trana_not_tranb(M, N, K, alpha, mat_a, lda, mat_b, ldb, mat_c, ldc);
  } else if (!transpose_a && transpose_b) {
    gemm_not_trana_tranb(M, N, K, alpha, mat_a, lda, mat_b, ldb, mat_c, ldc);
  } else {
    gemm_trana_not_tranb(M, N, K, alpha, mat_a, lda, mat_b, ldb, mat_c, ldc);
  }
}
