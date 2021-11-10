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
#if defined(ENABLE_SSE) && !defined(ENABLE_AVX)
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/fp32/common_func_fp32.h"

void WinogradTransLeft(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length) {
  size_t len_c4 = length * 4;
  size_t S_step = length * w * 4;
  for (int h1 = 0; h1 < h; ++h1) {
    const float *SW = S;
    memset(M, 0, len_c4 * w * sizeof(float));
    for (int w_tmp = w; w_tmp > 0; --w_tmp) {
      const float *SK = SW;
      const float *BK = B;
      int k_tmp = k;
      for (; k_tmp >= 7; k_tmp -= 7) {
        __m128 k1 = _mm_load_ps1(BK);
        __m128 k2 = _mm_load_ps1(BK + h);
        __m128 k3 = _mm_load_ps1(BK + 2 * h);
        __m128 k4 = _mm_load_ps1(BK + 3 * h);
        __m128 k5 = _mm_load_ps1(BK + 4 * h);
        __m128 k6 = _mm_load_ps1(BK + 5 * h);
        __m128 k7 = _mm_load_ps1(BK + 6 * h);
        BK += 7 * h;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, M += 4, SK += 4) {
#ifdef ENABLE_AVX
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_set1_ps(0.0f);
          __m128 s1 = _mm_loadu_ps(SK);
          M1 = _mm_fmadd_ps(s1, k1, M1);
          __m128 s2 = _mm_loadu_ps(SK + S_step);
          M2 = _mm_fmadd_ps(s2, k2, M2);
          __m128 s3 = _mm_loadu_ps(SK + 2 * S_step);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          __m128 s4 = _mm_loadu_ps(SK + 3 * S_step);
          M2 = _mm_fmadd_ps(s4, k4, M2);
          __m128 s5 = _mm_loadu_ps(SK + 4 * S_step);
          M1 = _mm_fmadd_ps(s5, k5, M1);
          __m128 s6 = _mm_loadu_ps(SK + 5 * S_step);
          M2 = _mm_fmadd_ps(s6, k6, M2);
          __m128 s7 = _mm_loadu_ps(SK + 6 * S_step);
          M1 = _mm_fmadd_ps(s7, k7, M1);
          M1 = _mm_add_ps(M1, M2);
          _mm_storeu_ps(M, M1);
#else
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
          __m128 s1 = _mm_loadu_ps(SK + S_step);
          s1 = _mm_mul_ps(s1, k2);
          __m128 s3 = _mm_loadu_ps(SK + 2 * S_step);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s3, k3));
          __m128 s4 = _mm_loadu_ps(SK + 3 * S_step);
          s1 = _mm_add_ps(s1, _mm_mul_ps(s4, k4));
          __m128 s5 = _mm_loadu_ps(SK + 4 * S_step);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s5, k5));
          __m128 s6 = _mm_loadu_ps(SK + 5 * S_step);
          s1 = _mm_add_ps(s1, _mm_mul_ps(s6, k6));
          __m128 s7 = _mm_loadu_ps(SK + 6 * S_step);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s7, k7));
          M1 = _mm_add_ps(M1, s1);
          _mm_storeu_ps(M, M1);
#endif
        }
        M -= len_c4;
        SK += 7 * S_step - len_c4;
      }
      for (; k_tmp >= 4; k_tmp -= 4) {
        __m128 k1 = _mm_load_ps1(BK);
        __m128 k2 = _mm_load_ps1(BK + h);
        __m128 k3 = _mm_load_ps1(BK + 2 * h);
        __m128 k4 = _mm_load_ps1(BK + 3 * h);
        BK += 4 * h;
        int len_tmp = length;
#ifdef ENABLE_AVX
        for (; len_tmp >= C2NUM; len_tmp -= C2NUM, SK += C8NUM, M += C8NUM) {
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_loadu_ps(M + C4NUM);
          __m128 s1 = _mm_loadu_ps(SK);
          __m128 s11 = _mm_loadu_ps(SK + C4NUM);
          M1 = _mm_fmadd_ps(s1, k1, M1);
          M2 = _mm_fmadd_ps(s11, k1, M2);
          __m128 s2 = _mm_loadu_ps(SK + S_step);
          __m128 s22 = _mm_loadu_ps(SK + S_step + C4NUM);
          M1 = _mm_fmadd_ps(s2, k2, M1);
          M2 = _mm_fmadd_ps(s22, k2, M2);
          __m128 s3 = _mm_loadu_ps(SK + 2 * S_step);
          __m128 s33 = _mm_loadu_ps(SK + 2 * S_step + C4NUM);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          M2 = _mm_fmadd_ps(s33, k3, M2);
          __m128 s4 = _mm_loadu_ps(SK + 3 * S_step);
          __m128 s44 = _mm_loadu_ps(SK + 3 * S_step + C4NUM);
          M1 = _mm_fmadd_ps(s4, k4, M1);
          M2 = _mm_fmadd_ps(s44, k4, M2);
          _mm_storeu_ps(M, M1);
          _mm_storeu_ps(M + C4NUM, M2);
        }
#endif
        for (; len_tmp > 0; --len_tmp, SK += 4, M += 4) {
#ifdef ENABLE_AVX
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_set1_ps(0.0f);
          __m128 s1 = _mm_loadu_ps(SK);
          M1 = _mm_fmadd_ps(s1, k1, M1);
          __m128 s2 = _mm_loadu_ps(SK + S_step);
          M2 = _mm_fmadd_ps(s2, k2, M2);
          __m128 s3 = _mm_loadu_ps(SK + 2 * S_step);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          __m128 s4 = _mm_loadu_ps(SK + 3 * S_step);
          M2 = _mm_fmadd_ps(s4, k4, M2);
          M1 = _mm_add_ps(M1, M2);
          _mm_storeu_ps(M, M1);
#else
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
          __m128 s1 = _mm_loadu_ps(SK + S_step);
          s1 = _mm_mul_ps(s1, k2);
          __m128 s3 = _mm_loadu_ps(SK + 2 * S_step);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s3, k3));
          __m128 s4 = _mm_loadu_ps(SK + 3 * S_step);
          s1 = _mm_add_ps(s1, _mm_mul_ps(s4, k4));
          M1 = _mm_add_ps(M1, s1);
          _mm_storeu_ps(M, M1);
#endif
        }
        M -= len_c4;
        SK += 4 * S_step - len_c4;
      }
      for (; k_tmp >= 3; k_tmp -= 3) {
        __m128 k1 = _mm_load_ps1(BK);
        __m128 k2 = _mm_load_ps1(BK + h);
        __m128 k3 = _mm_load_ps1(BK + 2 * h);
        BK += 3 * h;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, SK += 4, M += 4) {
#ifdef ENABLE_AVX
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_set1_ps(0.0f);
          __m128 s1 = _mm_loadu_ps(SK);
          M1 = _mm_fmadd_ps(s1, k1, M1);
          __m128 s2 = _mm_loadu_ps(SK + S_step);
          M2 = _mm_fmadd_ps(s2, k2, M2);
          __m128 s3 = _mm_loadu_ps(SK + 2 * S_step);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          M1 = _mm_add_ps(M1, M2);
          _mm_storeu_ps(M, M1);
#else
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
          __m128 s1 = _mm_loadu_ps(SK + S_step);
          s1 = _mm_mul_ps(s1, k2);
          __m128 s3 = _mm_loadu_ps(SK + 2 * S_step);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s3, k3));
          M1 = _mm_add_ps(M1, s1);
          _mm_storeu_ps(M, M1);
#endif
        }
        M -= len_c4;
        SK += 3 * S_step - len_c4;
      }
      for (; k_tmp > 0; k_tmp -= 1) {
        __m128 k1 = _mm_load_ps1(BK);
        BK += h;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, SK += 4, M += 4) {
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
#ifdef ENABLE_AVX
          M1 = _mm_fmadd_ps(s0, k1, M1);
#else
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
#endif
          _mm_storeu_ps(M, M1);
        }
        M -= len_c4;
        SK += S_step - len_c4;
      }
      SW += len_c4;
      M += len_c4;
    }
    B += 1;
  }
}

void WinogradTransRight(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length) {
  size_t len_c4 = length * 4, k_step = len_c4 * k;
  for (int h1 = 0; h1 < h; ++h1, S += k_step) {
    const float *BW = B;
    memset(M, 0, len_c4 * w * sizeof(float));
    for (int ww = 0; ww < w; ++ww, BW += 1, M += len_c4) {
      const float *SK = S, *BK = BW;
      int k_tmp = k;
      for (; k_tmp >= 7; k_tmp -= 7, M -= len_c4) {
        __m128 k1 = _mm_load_ps1(BK);
        __m128 k2 = _mm_load_ps1(BK + h);
        __m128 k3 = _mm_load_ps1(BK + 2 * h);
        __m128 k4 = _mm_load_ps1(BK + 3 * h);
        __m128 k5 = _mm_load_ps1(BK + 4 * h);
        __m128 k6 = _mm_load_ps1(BK + 5 * h);
        __m128 k7 = _mm_load_ps1(BK + 6 * h);
        BK += 7 * h;
        const float *S2 = SK + len_c4, *S3 = S2 + len_c4;
        const float *S4 = S3 + len_c4, *S5 = S4 + len_c4;
        const float *S6 = S5 + len_c4, *S7 = S6 + len_c4;
        for (int len_tmp = length; len_tmp > 0;
             --len_tmp, M += 4, SK += 4, S2 += 4, S3 += 4, S4 += 4, S5 += 4, S6 += 4, S7 += 4) {
#ifdef ENABLE_AVX
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_set1_ps(0.0f);
          __m128 s1 = _mm_loadu_ps(SK);
          M1 = _mm_fmadd_ps(s1, k1, M1);
          __m128 s2 = _mm_loadu_ps(S2);
          M2 = _mm_fmadd_ps(s2, k2, M2);
          __m128 s3 = _mm_loadu_ps(S3);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          __m128 s4 = _mm_loadu_ps(S4);
          M2 = _mm_fmadd_ps(s4, k4, M2);
          __m128 s5 = _mm_loadu_ps(S5);
          M1 = _mm_fmadd_ps(s5, k5, M1);
          __m128 s6 = _mm_loadu_ps(S6);
          M2 = _mm_fmadd_ps(s6, k6, M2);
          __m128 s7 = _mm_loadu_ps(S7);
          M1 = _mm_fmadd_ps(s7, k7, M1);
          M1 = _mm_add_ps(M1, M2);
          _mm_storeu_ps(M, M1);
#else
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
          __m128 s1 = _mm_loadu_ps(S2);
          s1 = _mm_mul_ps(s1, k2);
          __m128 s3 = _mm_loadu_ps(S3);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s3, k3));
          __m128 s4 = _mm_loadu_ps(S4);
          s1 = _mm_add_ps(s1, _mm_mul_ps(s4, k4));
          __m128 s5 = _mm_loadu_ps(S5);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s5, k5));
          __m128 s6 = _mm_loadu_ps(S6);
          s1 = _mm_add_ps(s1, _mm_mul_ps(s6, k6));
          __m128 s7 = _mm_loadu_ps(S7);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s7, k7));
          M1 = _mm_add_ps(M1, s1);
          _mm_storeu_ps(M, M1);
#endif
        }
        SK = S7;
      }
      for (; k_tmp >= 4; k_tmp -= 4, M -= len_c4) {
        __m128 k1 = _mm_load_ps1(BK);
        __m128 k2 = _mm_load_ps1(BK + h);
        __m128 k3 = _mm_load_ps1(BK + 2 * h);
        __m128 k4 = _mm_load_ps1(BK + 3 * h);
        BK += 4 * h;
        const float *S2 = SK + len_c4;
        const float *S3 = S2 + len_c4;
        const float *S4 = S3 + len_c4;
        int len_tmp = length;
#ifdef ENABLE_AVX
        for (; len_tmp >= C2NUM; len_tmp -= C2NUM, M += C8NUM, SK += C8NUM, S2 += C8NUM, S3 += C8NUM, S4 += C8NUM) {
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_loadu_ps(M + C4NUM);
          __m128 s1 = _mm_loadu_ps(SK);
          __m128 s11 = _mm_loadu_ps(SK + C4NUM);
          M1 = _mm_fmadd_ps(s1, k1, M1);
          M2 = _mm_fmadd_ps(s11, k1, M2);
          __m128 s2 = _mm_loadu_ps(S2);
          __m128 s22 = _mm_loadu_ps(S2 + C4NUM);
          M1 = _mm_fmadd_ps(s2, k2, M1);
          M2 = _mm_fmadd_ps(s22, k2, M2);
          __m128 s3 = _mm_loadu_ps(S3);
          __m128 s33 = _mm_loadu_ps(S3 + C4NUM);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          M2 = _mm_fmadd_ps(s33, k3, M2);
          __m128 s4 = _mm_loadu_ps(S4);
          __m128 s44 = _mm_loadu_ps(S4 + C4NUM);
          M1 = _mm_fmadd_ps(s4, k4, M1);
          M2 = _mm_fmadd_ps(s44, k4, M2);
          _mm_storeu_ps(M, M1);
          _mm_storeu_ps(M + C4NUM, M2);
        }
#endif
        for (; len_tmp > 0; --len_tmp, M += 4, SK += 4, S2 += 4, S3 += 4, S4 += 4) {
#ifdef ENABLE_AVX
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_set1_ps(0.0f);
          __m128 s1 = _mm_loadu_ps(SK);
          M1 = _mm_fmadd_ps(s1, k1, M1);
          __m128 s2 = _mm_loadu_ps(S2);
          M2 = _mm_fmadd_ps(s2, k2, M2);
          __m128 s3 = _mm_loadu_ps(S3);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          __m128 s4 = _mm_loadu_ps(S4);
          M2 = _mm_fmadd_ps(s4, k4, M2);
          M1 = _mm_add_ps(M1, M2);
          _mm_storeu_ps(M, M1);
#else
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
          __m128 s1 = _mm_loadu_ps(S2);
          s1 = _mm_mul_ps(s1, k2);
          __m128 s3 = _mm_loadu_ps(S3);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s3, k3));
          __m128 s4 = _mm_loadu_ps(S4);
          s1 = _mm_add_ps(s1, _mm_mul_ps(s4, k4));
          M1 = _mm_add_ps(M1, s1);
          _mm_storeu_ps(M, M1);
#endif
        }
        SK = S4;
      }
      for (; k_tmp >= 3; k_tmp -= 3, M -= len_c4) {
        __m128 k1 = _mm_load_ps1(BK);
        __m128 k2 = _mm_load_ps1(BK + h);
        __m128 k3 = _mm_load_ps1(BK + 2 * h);
        BK += 3 * h;
        const float *S2 = SK + len_c4;
        const float *S3 = S2 + len_c4;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, M += 4, SK += 4, S2 += 4, S3 += 4) {
#ifdef ENABLE_AVX
          __m128 M1 = _mm_loadu_ps(M);
          __m128 M2 = _mm_set1_ps(0.0f);
          __m128 s0 = _mm_loadu_ps(SK);
          M1 = _mm_fmadd_ps(s0, k1, M1);
          __m128 s1 = _mm_loadu_ps(S2);
          M2 = _mm_fmadd_ps(s1, k2, M2);
          __m128 s3 = _mm_loadu_ps(S3);
          M1 = _mm_fmadd_ps(s3, k3, M1);
          M1 = _mm_add_ps(M1, M2);
          _mm_storeu_ps(M, M1);
#else
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
          __m128 s1 = _mm_loadu_ps(S2);
          s1 = _mm_mul_ps(s1, k2);
          __m128 s3 = _mm_loadu_ps(S3);
          M1 = _mm_add_ps(M1, _mm_mul_ps(s3, k3));
          M1 = _mm_add_ps(M1, s1);
          _mm_storeu_ps(M, M1);
#endif
        }
        SK = S3;
      }
      for (; k_tmp >= 1; k_tmp -= 1, M -= len_c4) {
        __m128 k1 = _mm_load_ps1(BK);
        BK += h;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, M += 4, SK += 4) {
          __m128 M1 = _mm_loadu_ps(M);
          __m128 s0 = _mm_loadu_ps(SK);
#ifdef ENABLE_AVX
          M1 = _mm_fmadd_ps(s0, k1, M1);
#else
          M1 = _mm_add_ps(M1, _mm_mul_ps(s0, k1));
#endif
          _mm_storeu_ps(M, M1);
        }
      }
    }
  }
}
#endif
