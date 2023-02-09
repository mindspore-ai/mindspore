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
#ifndef MINDSPORE_NNACL_X86_64_AVX_COMMON_UTILS_H_
#define MINDSPORE_NNACL_X86_64_AVX_COMMON_UTILS_H_

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __GNUC__
#if __GNUC__ < 8
#define _mm256_set_m128i(xmm1, xmm2) \
  _mm256_permute2f128_si256(_mm256_castsi128_si256(xmm1), _mm256_castsi128_si256(xmm2), 2)
#define _mm256_set_m128f(xmm1, xmm2) \
  _mm256_permute2f128_ps(_mm256_castps128_ps256(xmm1), _mm256_castps128_ps256(xmm2), 2)
#endif
#endif

#define AVX_ACT_RELU 1
#define AVX_ACT_RELU6 3

// Signed saturating Add
__m128i _mm_adds_epi32(__m128i a, __m128i b);

// Signed rounding shift right
__m128i _mm_rshr_epi32(__m128i a, int shift);

// Signed saturating Rounding Doubling Multiply return High half
__m128i _mm_qrdmulh_epi32(__m128i a, __m128i b);

static inline void ActBlock1Avx(__m256 *v1, size_t relu, size_t relu6) {
  __m256 zero_ma = _mm256_setzero_ps();
  __m256 relu6_ma = _mm256_set1_ps(6.0f);
  if (relu || relu6) {
    *v1 = _mm256_max_ps(zero_ma, *v1);
  }
  if (relu6) {
    *v1 = _mm256_min_ps(relu6_ma, *v1);
  }
}

static inline void ActBlock2Avx(__m256 *v1, __m256 *v2, size_t relu, size_t relu6) {
  __m256 zero_ma = _mm256_setzero_ps();
  __m256 relu6_ma = _mm256_set1_ps(6.0f);
  if (relu || relu6) {
    *v1 = _mm256_max_ps(zero_ma, *v1);
    *v2 = _mm256_max_ps(zero_ma, *v2);
  }
  if (relu6) {
    *v1 = _mm256_min_ps(relu6_ma, *v1);
    *v2 = _mm256_min_ps(relu6_ma, *v2);
  }
}

static inline void ActBlock4Avx(__m256 *v1, __m256 *v2, __m256 *v3, __m256 *v4, size_t relu, size_t relu6) {
  __m256 zero_ma = _mm256_setzero_ps();
  __m256 relu6_ma = _mm256_set1_ps(6.0f);
  if (relu || relu6) {
    *v1 = _mm256_max_ps(zero_ma, *v1);
    *v2 = _mm256_max_ps(zero_ma, *v2);
    *v3 = _mm256_max_ps(zero_ma, *v3);
    *v4 = _mm256_max_ps(zero_ma, *v4);
  }
  if (relu6) {
    *v1 = _mm256_min_ps(relu6_ma, *v1);
    *v2 = _mm256_min_ps(relu6_ma, *v2);
    *v3 = _mm256_min_ps(relu6_ma, *v3);
    *v4 = _mm256_min_ps(relu6_ma, *v4);
  }
}

static inline void ActBlock8Avx(__m256 *v1, __m256 *v2, __m256 *v3, __m256 *v4, __m256 *v5, __m256 *v6, __m256 *v7,
                                __m256 *v8, size_t relu_type) {
  __m256 relu6 = _mm256_set1_ps(6.0);
  __m256 zero = _mm256_setzero_ps();
  switch (relu_type) {
    case AVX_ACT_RELU6:
      *v1 = _mm256_min_ps(*v1, relu6);
      *v2 = _mm256_min_ps(*v2, relu6);
      *v3 = _mm256_min_ps(*v3, relu6);
      *v4 = _mm256_min_ps(*v4, relu6);
      *v5 = _mm256_min_ps(*v5, relu6);
      *v6 = _mm256_min_ps(*v6, relu6);
      *v7 = _mm256_min_ps(*v7, relu6);
      *v8 = _mm256_min_ps(*v8, relu6);
    case AVX_ACT_RELU:
      *v1 = _mm256_max_ps(*v1, zero);
      *v2 = _mm256_max_ps(*v2, zero);
      *v3 = _mm256_max_ps(*v3, zero);
      *v4 = _mm256_max_ps(*v4, zero);
      *v5 = _mm256_max_ps(*v5, zero);
      *v6 = _mm256_max_ps(*v6, zero);
      *v7 = _mm256_max_ps(*v7, zero);
      *v8 = _mm256_max_ps(*v8, zero);
      break;
    default:
      break;
  }
}

static inline void ActBlock12Avx(__m256 *v1, __m256 *v2, __m256 *v3, __m256 *v4, __m256 *v5, __m256 *v6, __m256 *v7,
                                 __m256 *v8, __m256 *v9, __m256 *v10, __m256 *v11, __m256 *v12, size_t relu,
                                 size_t relu6) {
  if (relu || relu6) {
    __m256 zero_ma = _mm256_setzero_ps();
    *v1 = _mm256_max_ps(zero_ma, *v1);
    *v2 = _mm256_max_ps(zero_ma, *v2);
    *v3 = _mm256_max_ps(zero_ma, *v3);
    *v4 = _mm256_max_ps(zero_ma, *v4);
    *v5 = _mm256_max_ps(zero_ma, *v5);
    *v6 = _mm256_max_ps(zero_ma, *v6);
    *v7 = _mm256_max_ps(zero_ma, *v7);
    *v8 = _mm256_max_ps(zero_ma, *v8);
    *v9 = _mm256_max_ps(zero_ma, *v9);
    *v10 = _mm256_max_ps(zero_ma, *v10);
    *v11 = _mm256_max_ps(zero_ma, *v11);
    *v12 = _mm256_max_ps(zero_ma, *v12);
  }
  if (relu6) {
    __m256 relu6_ma = _mm256_set1_ps(6.0f);
    *v1 = _mm256_min_ps(relu6_ma, *v1);
    *v2 = _mm256_min_ps(relu6_ma, *v2);
    *v3 = _mm256_min_ps(relu6_ma, *v3);
    *v4 = _mm256_min_ps(relu6_ma, *v4);
    *v5 = _mm256_min_ps(relu6_ma, *v5);
    *v6 = _mm256_min_ps(relu6_ma, *v6);
    *v7 = _mm256_min_ps(relu6_ma, *v7);
    *v8 = _mm256_min_ps(relu6_ma, *v8);
    *v9 = _mm256_min_ps(relu6_ma, *v9);
    *v10 = _mm256_min_ps(relu6_ma, *v10);
    *v11 = _mm256_min_ps(relu6_ma, *v11);
    *v12 = _mm256_min_ps(relu6_ma, *v12);
  }
}

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_X86_64_AVX_COMMON_UTILS_H_
