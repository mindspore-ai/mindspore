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

static void gemm_nn(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_B, int ldb, float *mat_c,
                    int ldc) {
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float a = alpha * mat_a[i * lda + k];
      for (j = 0; j < N; ++j) {
        mat_c[i * ldc + j] += a * mat_B[k * ldb + j];
      }
    }
  }
}

static void gemm_nt(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b, int ldb, float *mat_c,
                    int ldc) {
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float sum = 0;
      for (k = 0; k < K; ++k) {
        sum += alpha * mat_a[i * lda + k] * mat_b[j * ldb + k];
      }
      mat_c[i * ldc + j] += sum;
    }
  }
}

static void gemm_tn(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b, int ldb, float *mat_c,
                    int ldc) {
  int i, j, k;
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      float a = alpha * mat_a[k * lda + i];
      for (j = 0; j < N; ++j) {
        mat_c[i * ldc + j] += a * mat_b[k * ldb + j];
      }
    }
  }
}

static void gemm_tt(int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b, int ldb, float *mat_c,
                    int ldc) {
  int i, j, k;
  for (i = 0; i < M; ++i) {
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

void gemm(int transpose_a, int transpose_b, int M, int N, int K, float alpha, float *mat_a, int lda, float *mat_b,
          int ldb, float beta, float *mat_c, int ldc) {
  // printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
  if (beta >= 0.f && beta <= 0.f) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        mat_c[i * ldc + j] = 0;
      }
    }
  } else if (beta < 1.f || beta > 1.f) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        mat_c[i * ldc + j] *= beta;
      }
    }
  }

  int t;

  for (t = 0; t < M; ++t) {
    if (!transpose_a && !transpose_b) {
      gemm_nn(1, N, K, alpha, mat_a + t * lda, lda, mat_b, ldb, mat_c + t * ldc, ldc);
    } else if (transpose_a && !transpose_b) {
      gemm_tn(1, N, K, alpha, mat_a + t, lda, mat_b, ldb, mat_c + t * ldc, ldc);
    } else if (!transpose_a && transpose_b) {
      gemm_nt(1, N, K, alpha, mat_a + t * lda, lda, mat_b, ldb, mat_c + t * ldc, ldc);
    } else {
      gemm_tt(1, N, K, alpha, mat_a + t, lda, mat_b, ldb, mat_c + t * ldc, ldc);
    }
  }
}
