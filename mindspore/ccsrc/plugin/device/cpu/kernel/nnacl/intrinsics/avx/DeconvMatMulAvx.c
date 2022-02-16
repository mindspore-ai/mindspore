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
#ifdef ENABLE_AVX
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/op_base.h"

void Deconv4X8AvxKernel(const float *src, const float *weight, float *dst, int col, int row, int depth, int stride) {
  __m256 res1 = _mm256_setzero_ps();
  __m256 res4 = _mm256_setzero_ps();
  __m256 res7 = _mm256_setzero_ps();
  __m256 res10 = _mm256_setzero_ps();

  for (int d = 0; d < depth; ++d) {
    __m256 w0 = _mm256_loadu_ps(weight);
    __m256 tmp = _mm256_set1_ps(*src);
    __m256 tmp1 = _mm256_set1_ps(*(src + C1NUM));
    weight += C8NUM;
    __m256 tmp2 = _mm256_set1_ps(*(src + C2NUM));
    __m256 tmp3 = _mm256_set1_ps(*(src + C3NUM));
    res1 = _mm256_fmadd_ps(tmp, w0, res1);
    res4 = _mm256_fmadd_ps(tmp1, w0, res4);
    src += C4NUM;
    res7 = _mm256_fmadd_ps(tmp2, w0, res7);
    res10 = _mm256_fmadd_ps(tmp3, w0, res10);
  }
  // write
  _mm256_storeu_ps(dst, res1);
  _mm256_storeu_ps(dst + C8NUM, res4);
  _mm256_storeu_ps(dst + C16NUM, res7);
  _mm256_storeu_ps(dst + C24NUM, res10);
}

void Deconv4X16AvxKernel(const float *src, const float *weight, float *dst, int col, int row, int depth, int stride) {
  __m256 res1 = _mm256_setzero_ps();
  __m256 res2 = _mm256_setzero_ps();
  __m256 res4 = _mm256_setzero_ps();
  __m256 res5 = _mm256_setzero_ps();
  __m256 res7 = _mm256_setzero_ps();
  __m256 res8 = _mm256_setzero_ps();
  __m256 res10 = _mm256_setzero_ps();
  __m256 res11 = _mm256_setzero_ps();

  for (int d = 0; d < depth; ++d) {
    __m256 w0 = _mm256_loadu_ps(weight);
    __m256 w1 = _mm256_loadu_ps(weight + C8NUM);
    weight += C16NUM;
    __m256 tmp = _mm256_set1_ps(*src);
    __m256 tmp1 = _mm256_set1_ps(*(src + C1NUM));
    __m256 tmp2 = _mm256_set1_ps(*(src + C2NUM));
    __m256 tmp3 = _mm256_set1_ps(*(src + C3NUM));
    res1 = _mm256_fmadd_ps(tmp, w0, res1);
    res2 = _mm256_fmadd_ps(tmp, w1, res2);
    src += C4NUM;
    res4 = _mm256_fmadd_ps(tmp1, w0, res4);
    res5 = _mm256_fmadd_ps(tmp1, w1, res5);
    res7 = _mm256_fmadd_ps(tmp2, w0, res7);
    res8 = _mm256_fmadd_ps(tmp2, w1, res8);
    res10 = _mm256_fmadd_ps(tmp3, w0, res10);
    res11 = _mm256_fmadd_ps(tmp3, w1, res11);
  }
  // write
  _mm256_storeu_ps(dst, res1);
  _mm256_storeu_ps(dst + C8NUM, res4);
  _mm256_storeu_ps(dst + C16NUM, res7);
  _mm256_storeu_ps(dst + C24NUM, res10);

  _mm256_storeu_ps(dst + stride, res2);
  _mm256_storeu_ps(dst + stride + C8NUM, res5);
  _mm256_storeu_ps(dst + stride + C16NUM, res8);
  _mm256_storeu_ps(dst + stride + C24NUM, res11);
}

void Deconv4X24AvxKernel(const float *src, const float *weight, float *dst, int col, int row, int depth, int stride) {
  __m256 res1 = _mm256_setzero_ps();
  __m256 res2 = _mm256_setzero_ps();
  __m256 res3 = _mm256_setzero_ps();
  __m256 res4 = _mm256_setzero_ps();
  __m256 res5 = _mm256_setzero_ps();
  __m256 res6 = _mm256_setzero_ps();
  __m256 res7 = _mm256_setzero_ps();
  __m256 res8 = _mm256_setzero_ps();
  __m256 res9 = _mm256_setzero_ps();
  __m256 res10 = _mm256_setzero_ps();
  __m256 res11 = _mm256_setzero_ps();
  __m256 res12 = _mm256_setzero_ps();

  for (int d = 0; d < depth; ++d) {
    __m256 w0 = _mm256_loadu_ps(weight);
    __m256 w1 = _mm256_loadu_ps(weight + C8NUM);
    __m256 w2 = _mm256_loadu_ps(weight + C16NUM);
    __m256 tmp = _mm256_set1_ps(*src);
    res1 = _mm256_fmadd_ps(tmp, w0, res1);
    res2 = _mm256_fmadd_ps(tmp, w1, res2);
    res3 = _mm256_fmadd_ps(tmp, w2, res3);
    tmp = _mm256_set1_ps(*(src + C1NUM));
    res4 = _mm256_fmadd_ps(tmp, w0, res4);
    res5 = _mm256_fmadd_ps(tmp, w1, res5);
    res6 = _mm256_fmadd_ps(tmp, w2, res6);
    tmp = _mm256_set1_ps(*(src + C2NUM));
    res7 = _mm256_fmadd_ps(tmp, w0, res7);
    res8 = _mm256_fmadd_ps(tmp, w1, res8);
    res9 = _mm256_fmadd_ps(tmp, w2, res9);
    tmp = _mm256_set1_ps(*(src + C3NUM));
    res10 = _mm256_fmadd_ps(tmp, w0, res10);
    res11 = _mm256_fmadd_ps(tmp, w1, res11);
    res12 = _mm256_fmadd_ps(tmp, w2, res12);
    weight += C24NUM;
    src += C4NUM;
  }
  // write
  _mm256_storeu_ps(dst, res1);
  _mm256_storeu_ps(dst + C8NUM, res4);
  _mm256_storeu_ps(dst + C16NUM, res7);
  _mm256_storeu_ps(dst + C24NUM, res10);

  _mm256_storeu_ps(dst + stride, res2);
  _mm256_storeu_ps(dst + stride + C8NUM, res5);
  _mm256_storeu_ps(dst + stride + C16NUM, res8);
  _mm256_storeu_ps(dst + stride + C24NUM, res11);

  _mm256_storeu_ps(dst + C2NUM * stride, res3);
  _mm256_storeu_ps(dst + C2NUM * stride + C8NUM, res6);
  _mm256_storeu_ps(dst + C2NUM * stride + C16NUM, res9);
  _mm256_storeu_ps(dst + C2NUM * stride + C24NUM, res12);
}

void DeconvMatmulAvx(const float *a, const float *b, float *c, int depth, int row, int col, const int plane) {
  NNACL_CHECK_ZERO_RETURN(plane);
  int col_num = 0;
  int col_block = UP_DIV(col / plane, C8NUM);
  DeconvAvxKernel kernel[3] = {Deconv4X8AvxKernel, Deconv4X16AvxKernel, Deconv4X24AvxKernel};
  for (int col_tmp = 0; col_tmp < col_block; col_tmp += col_num) {
    col_num = MSMIN(C3NUM, col_block - col_tmp);
    for (int p = 0; p < plane; ++p) {
      for (int r = 0; r < row; r += C4NUM) {
        kernel[col_num - 1](a + r * depth, b + (col_tmp * plane + p * col_num) * C8NUM * depth,
                            c + (col_tmp * plane + p) * C8NUM * row + r * C8NUM, col_num, C4NUM, depth,
                            row * C8NUM * plane);
      }
    }
  }
}

#ifdef ENABLE_DEBUG
void DeconvColXRowAvxKernel(const float *src, const float *weight, float *dst, int col, int row, int depth,
                            int stride) {
  __m256 res[C12NUM];
  __m256 w[C3NUM];
  for (int i = 0; i < C12NUM; ++i) {
    res[i] = _mm256_setzero_ps();
  }
  for (int d = 0; d < depth; ++d) {
    for (int c = 0; c < col; ++c) {
      w[c] = _mm256_loadu_ps(weight + c * C8NUM);
    }
    weight += col * C8NUM;
    for (int r = 0; r < row; ++r) {  // C4NUm
      __m256 tmp = _mm256_set1_ps(*src);
      for (int c = 0; c < col; ++c) {  // 3 * C8NUM
        res[r * col + c] = _mm256_fmadd_ps(tmp, w[c], res[r * col + c]);
      }
      src += 1;
    }
  }
  // write
  for (int i = 0; i < col; ++i) {
    for (int j = 0; j < row; ++j) {
      _mm256_storeu_ps(dst + j * C8NUM, res[j * col + i]);
    }
    dst += stride;
  }
}
#endif
#endif
