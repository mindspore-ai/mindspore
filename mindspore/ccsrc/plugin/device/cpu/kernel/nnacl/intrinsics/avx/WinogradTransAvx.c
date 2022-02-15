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
#ifdef ENABLE_AVX
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/fp32/common_func_fp32.h"

void WinogradTransLeft(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length) {
  size_t len_c8 = length * C8NUM;
  size_t S_step = length * w * C8NUM;
  for (int h1 = 0; h1 < h; ++h1) {
    const float *SW = S;
    memset(M, 0, len_c8 * w * sizeof(float));
    for (int w_tmp = w; w_tmp > 0; --w_tmp) {
      const float *SK = SW;
      const float *BK = B;
      int k_tmp = k;
      for (; k_tmp >= C8NUM; k_tmp -= C8NUM) {
        __m256 k1 = _mm256_set1_ps(*BK);
        __m256 k2 = _mm256_set1_ps(*(BK + h));
        __m256 k3 = _mm256_set1_ps(*(BK + C2NUM * h));
        __m256 k4 = _mm256_set1_ps(*(BK + C3NUM * h));
        __m256 k5 = _mm256_set1_ps(*(BK + C4NUM * h));
        __m256 k6 = _mm256_set1_ps(*(BK + C5NUM * h));
        __m256 k7 = _mm256_set1_ps(*(BK + C6NUM * h));
        __m256 k8 = _mm256_set1_ps(*(BK + C7NUM * h));
        BK += C8NUM * h;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, M += C8NUM, SK += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          __m256 s2 = _mm256_loadu_ps(SK + S_step);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          __m256 s3 = _mm256_loadu_ps(SK + C2NUM * S_step);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          __m256 s4 = _mm256_loadu_ps(SK + C3NUM * S_step);
          M2 = _mm256_fmadd_ps(s4, k4, M2);
          __m256 s5 = _mm256_loadu_ps(SK + C4NUM * S_step);
          M1 = _mm256_fmadd_ps(s5, k5, M1);
          __m256 s6 = _mm256_loadu_ps(SK + C5NUM * S_step);
          M2 = _mm256_fmadd_ps(s6, k6, M2);
          __m256 s7 = _mm256_loadu_ps(SK + C6NUM * S_step);
          M1 = _mm256_fmadd_ps(s7, k7, M1);
          __m256 s8 = _mm256_loadu_ps(SK + C7NUM * S_step);
          M2 = _mm256_fmadd_ps(s8, k8, M2);
          M1 = _mm256_add_ps(M1, M2);
          _mm256_storeu_ps(M, M1);
        }
        M -= len_c8;
        SK += C8NUM * S_step - len_c8;
      }
      for (; k_tmp >= C4NUM; k_tmp -= C4NUM) {
        __m256 k1 = _mm256_set1_ps(*BK);
        __m256 k2 = _mm256_set1_ps(*(BK + h));
        __m256 k3 = _mm256_set1_ps(*(BK + C2NUM * h));
        __m256 k4 = _mm256_set1_ps(*(BK + C3NUM * h));
        BK += C4NUM * h;
        int len_tmp = length;
        for (; len_tmp >= C2NUM; len_tmp -= C2NUM, SK += C16NUM, M += C16NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 M3 = _mm256_loadu_ps(M + C8NUM);
          __m256 M4 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          __m256 s11 = _mm256_loadu_ps(SK + C8NUM);
          __m256 s2 = _mm256_loadu_ps(SK + S_step);
          __m256 s22 = _mm256_loadu_ps(SK + S_step + C8NUM);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          M3 = _mm256_fmadd_ps(s11, k1, M3);
          M4 = _mm256_fmadd_ps(s22, k2, M4);
          __m256 s3 = _mm256_loadu_ps(SK + C2NUM * S_step);
          __m256 s33 = _mm256_loadu_ps(SK + C2NUM * S_step + C8NUM);
          __m256 s4 = _mm256_loadu_ps(SK + C3NUM * S_step);
          __m256 s44 = _mm256_loadu_ps(SK + C3NUM * S_step + C8NUM);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          M2 = _mm256_fmadd_ps(s4, k4, M2);
          M3 = _mm256_fmadd_ps(s33, k3, M3);
          M4 = _mm256_fmadd_ps(s44, k4, M4);
          M1 = _mm256_add_ps(M1, M2);
          M4 = _mm256_add_ps(M3, M4);
          _mm256_storeu_ps(M, M1);
          _mm256_storeu_ps(M + C8NUM, M4);
        }
        for (; len_tmp > 0; len_tmp--, SK += C8NUM, M += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          __m256 s2 = _mm256_loadu_ps(SK + S_step);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          __m256 s3 = _mm256_loadu_ps(SK + C2NUM * S_step);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          __m256 s4 = _mm256_loadu_ps(SK + C3NUM * S_step);
          M2 = _mm256_fmadd_ps(s4, k4, M2);
          M1 = _mm256_add_ps(M1, M2);
          _mm256_storeu_ps(M, M1);
        }
        M -= len_c8;
        SK += C4NUM * S_step - len_c8;
      }
      for (; k_tmp >= C3NUM; k_tmp -= C3NUM) {
        __m256 k1 = _mm256_set1_ps(*BK);
        __m256 k2 = _mm256_set1_ps(*(BK + h));
        __m256 k3 = _mm256_set1_ps(*(BK + C2NUM * h));
        BK += C3NUM * h;
        int len_tmp = length;
        for (; len_tmp >= C3NUM; len_tmp -= C3NUM, SK += C24NUM, M += C24NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 M3 = _mm256_loadu_ps(M + C8NUM);
          __m256 M4 = _mm256_set1_ps(0.0f);
          __m256 M5 = _mm256_loadu_ps(M + C16NUM);
          __m256 M6 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          __m256 s2 = _mm256_loadu_ps(SK + S_step);
          __m256 s11 = _mm256_loadu_ps(SK + C8NUM);
          __m256 s22 = _mm256_loadu_ps(SK + S_step + C8NUM);
          __m256 s111 = _mm256_loadu_ps(SK + C16NUM);
          __m256 s222 = _mm256_loadu_ps(SK + S_step + C16NUM);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          M3 = _mm256_fmadd_ps(s11, k1, M3);
          M4 = _mm256_fmadd_ps(s22, k2, M4);
          M5 = _mm256_fmadd_ps(s111, k1, M5);
          M6 = _mm256_fmadd_ps(s222, k2, M6);
          __m256 s3 = _mm256_loadu_ps(SK + C2NUM * S_step);
          __m256 s33 = _mm256_loadu_ps(SK + C2NUM * S_step + C8NUM);
          __m256 s333 = _mm256_loadu_ps(SK + C2NUM * S_step + C16NUM);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          M3 = _mm256_fmadd_ps(s33, k3, M3);
          M5 = _mm256_fmadd_ps(s333, k3, M5);
          M1 = _mm256_add_ps(M1, M2);
          M3 = _mm256_add_ps(M3, M4);
          M5 = _mm256_add_ps(M5, M6);
          _mm256_storeu_ps(M, M1);
          _mm256_storeu_ps(M + C8NUM, M3);
          _mm256_storeu_ps(M + C16NUM, M5);
        }
        for (; len_tmp > 0; len_tmp--, SK += C8NUM, M += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          __m256 s2 = _mm256_loadu_ps(SK + S_step);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          __m256 s3 = _mm256_loadu_ps(SK + C2NUM * S_step);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          M1 = _mm256_add_ps(M1, M2);
          _mm256_storeu_ps(M, M1);
        }
        M -= len_c8;
        SK += C3NUM * S_step - len_c8;
      }
      for (; k_tmp > 0; k_tmp -= 1) {
        __m256 k1 = _mm256_set1_ps(*BK);
        BK += h;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, SK += C8NUM, M += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 s0 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s0, k1, M1);
          _mm256_storeu_ps(M, M1);
        }
        M -= len_c8;
        SK += S_step - len_c8;
      }
      SW += len_c8;
      M += len_c8;
    }
    B += 1;
  }
}

void WinogradTransRight(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length) {
  size_t len_c8 = length * C8NUM, k_step = len_c8 * k;
  for (int h1 = 0; h1 < h; ++h1, S += k_step) {
    const float *BW = B;
    memset(M, 0, len_c8 * w * sizeof(float));
    for (int ww = 0; ww < w; ++ww, BW += 1, M += len_c8) {
      const float *SK = S, *BK = BW;
      int k_tmp = k;
      for (; k_tmp >= C8NUM; k_tmp -= C8NUM, M -= len_c8) {
        __m256 k1 = _mm256_set1_ps(*BK);
        __m256 k2 = _mm256_set1_ps(*(BK + h));
        __m256 k3 = _mm256_set1_ps(*(BK + C2NUM * h));
        __m256 k4 = _mm256_set1_ps(*(BK + C3NUM * h));
        __m256 k5 = _mm256_set1_ps(*(BK + C4NUM * h));
        __m256 k6 = _mm256_set1_ps(*(BK + C5NUM * h));
        __m256 k7 = _mm256_set1_ps(*(BK + C6NUM * h));
        __m256 k8 = _mm256_set1_ps(*(BK + C7NUM * h));
        BK += C8NUM * h;
        const float *S2 = SK + len_c8, *S3 = S2 + len_c8;
        const float *S4 = S3 + len_c8, *S5 = S4 + len_c8;
        const float *S6 = S5 + len_c8, *S7 = S6 + len_c8, *S8 = S7 + len_c8;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, M += C8NUM, SK += C8NUM, S2 += C8NUM, S3 += C8NUM,
                 S4 += C8NUM, S5 += C8NUM, S6 += C8NUM, S7 += C8NUM, S8 += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          __m256 s2 = _mm256_loadu_ps(S2);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          __m256 s3 = _mm256_loadu_ps(S3);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          __m256 s4 = _mm256_loadu_ps(S4);
          M2 = _mm256_fmadd_ps(s4, k4, M2);
          __m256 s5 = _mm256_loadu_ps(S5);
          M1 = _mm256_fmadd_ps(s5, k5, M1);
          __m256 s6 = _mm256_loadu_ps(S6);
          M2 = _mm256_fmadd_ps(s6, k6, M2);
          __m256 s7 = _mm256_loadu_ps(S7);
          M1 = _mm256_fmadd_ps(s7, k7, M1);
          __m256 s8 = _mm256_loadu_ps(S8);
          M2 = _mm256_fmadd_ps(s8, k8, M2);
          M1 = _mm256_add_ps(M1, M2);
          _mm256_storeu_ps(M, M1);
        }
        SK = S7;
      }
      for (; k_tmp >= C4NUM; k_tmp -= C4NUM, M -= len_c8) {
        __m256 k1 = _mm256_set1_ps(*BK);
        __m256 k2 = _mm256_set1_ps(*(BK + h));
        __m256 k3 = _mm256_set1_ps(*(BK + C2NUM * h));
        __m256 k4 = _mm256_set1_ps(*(BK + C3NUM * h));
        BK += C4NUM * h;
        const float *S2 = SK + len_c8;
        const float *S3 = S2 + len_c8;
        const float *S4 = S3 + len_c8;
        int len_tmp = length;
        for (; len_tmp >= C2NUM;
             len_tmp -= C2NUM, M += C16NUM, SK += C16NUM, S2 += C16NUM, S3 += C16NUM, S4 += C16NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 M3 = _mm256_loadu_ps(M + C8NUM);
          __m256 M4 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          __m256 s2 = _mm256_loadu_ps(S2);
          __m256 s11 = _mm256_loadu_ps(SK + C8NUM);
          __m256 s22 = _mm256_loadu_ps(S2 + C8NUM);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          M3 = _mm256_fmadd_ps(s11, k1, M3);
          M4 = _mm256_fmadd_ps(s22, k2, M4);

          __m256 s3 = _mm256_loadu_ps(S3);
          __m256 s4 = _mm256_loadu_ps(S4);
          __m256 s33 = _mm256_loadu_ps(S3 + C8NUM);
          __m256 s44 = _mm256_loadu_ps(S4 + C8NUM);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          M2 = _mm256_fmadd_ps(s4, k4, M2);
          M3 = _mm256_fmadd_ps(s33, k3, M3);
          M4 = _mm256_fmadd_ps(s44, k4, M4);

          M1 = _mm256_add_ps(M1, M2);
          M3 = _mm256_add_ps(M3, M4);
          _mm256_storeu_ps(M, M1);
          _mm256_storeu_ps(M + C8NUM, M3);
        }
        for (; len_tmp > 0; len_tmp--, M += C8NUM, SK += C8NUM, S2 += C8NUM, S3 += C8NUM, S4 += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          __m256 s2 = _mm256_loadu_ps(S2);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          __m256 s3 = _mm256_loadu_ps(S3);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          __m256 s4 = _mm256_loadu_ps(S4);
          M2 = _mm256_fmadd_ps(s4, k4, M2);
          M1 = _mm256_add_ps(M1, M2);
          _mm256_storeu_ps(M, M1);
        }
        SK = S4;
      }
      for (; k_tmp >= C3NUM; k_tmp -= C3NUM, M -= len_c8) {
        __m256 k1 = _mm256_set1_ps(*BK);
        __m256 k2 = _mm256_set1_ps(*(BK + h));
        __m256 k3 = _mm256_set1_ps(*(BK + C2NUM * h));
        BK += C3NUM * h;
        const float *S2 = SK + len_c8;
        const float *S3 = S2 + len_c8;
        int len_tmp = length;
        for (; len_tmp >= C3NUM; len_tmp -= C3NUM, M += C24NUM, SK += C24NUM, S2 += C24NUM, S3 += C24NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 M3 = _mm256_loadu_ps(M + C8NUM);
          __m256 M4 = _mm256_set1_ps(0.0f);
          __m256 M5 = _mm256_loadu_ps(M + C16NUM);
          __m256 M6 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          __m256 s2 = _mm256_loadu_ps(S2);
          __m256 s11 = _mm256_loadu_ps(SK + C8NUM);
          __m256 s22 = _mm256_loadu_ps(S2 + C8NUM);
          __m256 s111 = _mm256_loadu_ps(SK + C16NUM);
          __m256 s222 = _mm256_loadu_ps(S2 + C16NUM);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          M3 = _mm256_fmadd_ps(s11, k1, M3);
          M4 = _mm256_fmadd_ps(s22, k2, M4);
          M5 = _mm256_fmadd_ps(s111, k1, M5);
          M6 = _mm256_fmadd_ps(s222, k2, M6);
          __m256 s3 = _mm256_loadu_ps(S3);
          __m256 s33 = _mm256_loadu_ps(S3 + C8NUM);
          __m256 s333 = _mm256_loadu_ps(S3 + C16NUM);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          M3 = _mm256_fmadd_ps(s33, k3, M3);
          M5 = _mm256_fmadd_ps(s333, k3, M5);
          M1 = _mm256_add_ps(M1, M2);
          M3 = _mm256_add_ps(M3, M4);
          M5 = _mm256_add_ps(M6, M5);
          _mm256_storeu_ps(M, M1);
          _mm256_storeu_ps(M + C8NUM, M3);
          _mm256_storeu_ps(M + C16NUM, M5);
        }
        for (; len_tmp > 0; len_tmp--, M += C8NUM, SK += C8NUM, S2 += C8NUM, S3 += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 M2 = _mm256_set1_ps(0.0f);
          __m256 s1 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          __m256 s2 = _mm256_loadu_ps(S2);
          M2 = _mm256_fmadd_ps(s2, k2, M2);
          __m256 s3 = _mm256_loadu_ps(S3);
          M1 = _mm256_fmadd_ps(s3, k3, M1);
          M1 = _mm256_add_ps(M1, M2);
          _mm256_storeu_ps(M, M1);
        }
        SK = S3;
      }
      for (; k_tmp >= 1; k_tmp -= 1, M -= len_c8) {
        __m256 k1 = _mm256_set1_ps(*BK);
        BK += h;
        for (int len_tmp = length; len_tmp > 0; --len_tmp, M += C8NUM, SK += C8NUM) {
          __m256 M1 = _mm256_loadu_ps(M);
          __m256 s1 = _mm256_loadu_ps(SK);
          M1 = _mm256_fmadd_ps(s1, k1, M1);
          _mm256_storeu_ps(M, M1);
        }
      }
    }
  }
}
#endif
