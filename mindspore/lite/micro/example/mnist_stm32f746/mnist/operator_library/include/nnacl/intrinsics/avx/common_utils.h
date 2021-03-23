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
#ifndef MINDSPORE_LITE_NNACL_X86_64_AVX_COMMON_UTILS_H_
#define MINDSPORE_LITE_NNACL_X86_64_AVX_COMMON_UTILS_H_

#include <x86intrin.h>

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

// Signed saturating Add
__m128i _mm_adds_epi32(__m128i a, __m128i b);

// Signed rounding shift right
__m128i _mm_rshr_epi32(__m128i a, int shift);

// Signed saturating Rounding Doubling Multiply return High half
__m128i _mm_qrdmulh_epi32(__m128i a, __m128i b);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_NNACL_X86_64_AVX_COMMON_UTILS_H_
