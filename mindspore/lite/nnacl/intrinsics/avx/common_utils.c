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
#include "nnacl/intrinsics/avx/common_utils.h"
#ifdef WIN32
#ifdef ENABLE_AVX
#include <stdint.h>
#endif
#endif

__m128i _mm_adds_epi32(__m128i a, __m128i b) {
  __m128i int_min = _mm_set1_epi32(0x80000000);
  __m128i int_max = _mm_set1_epi32(0x7FFFFFFF);

  const __m128i res = _mm_add_epi32(a, b);
  const __m128i sign_and = _mm_and_si128(a, b);
  const __m128i sign_or = _mm_or_si128(a, b);

  const __m128i min_sat_mask = _mm_andnot_si128(res, sign_and);
  const __m128i max_sat_mask = _mm_andnot_si128(sign_or, res);
  const __m128 res_temp =
    _mm_blendv_ps(_mm_castsi128_ps(res), _mm_castsi128_ps(int_min), _mm_castsi128_ps(min_sat_mask));
  return _mm_castps_si128(_mm_blendv_ps(res_temp, _mm_castsi128_ps(int_max), _mm_castsi128_ps(max_sat_mask)));
}

__m128i _mm_rshr_epi32(__m128i a, int shift) {
  const __m128i vmask = _mm_cmpgt_epi32(_mm_setzero_si128(), a);
  const __m128i vabs_a = _mm_sub_epi32(_mm_xor_si128(a, vmask), vmask);
  const __m128i tmp_res = _mm_srli_epi32(vabs_a, shift);
  return _mm_xor_si128(tmp_res, vmask);
}

__m128i _mm_qrdmulh_epi32(__m128i a, __m128i b) {
  const __m128i tmp_a_lo = _mm_unpacklo_epi32(a, _mm_setzero_si128());
  const __m128i tmp_a_hi = _mm_unpackhi_epi32(a, _mm_setzero_si128());
  const __m256i tmp_a_256 = _mm256_set_m128i(tmp_a_hi, tmp_a_lo);
  const __m128i tmp_b_lo = _mm_unpacklo_epi32(b, _mm_setzero_si128());
  const __m128i tmp_b_hi = _mm_unpackhi_epi32(b, _mm_setzero_si128());
  const __m256i tmp_b_256 = _mm256_set_m128i(tmp_b_hi, tmp_b_lo);
  __m256i tmp_out = _mm256_mul_epi32(tmp_a_256, tmp_b_256);
  tmp_out = _mm256_add_epi64(tmp_out, _mm256_set1_epi64x(1ll << 30));
  const __m256i vmask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), tmp_out);
  const __m256i vabs_tmp_out = _mm256_sub_epi64(_mm256_xor_si256(tmp_out, vmask), vmask);
  tmp_out = _mm256_srli_epi64(vabs_tmp_out, 31);
  const __m256i vtmp_out = _mm256_sub_epi64(_mm256_xor_si256(tmp_out, vmask), vmask);
  const int32_t max_32bit = (1ll << 31) - 1;
  const int32_t min_32bit = -(1ll << 31);
  int64_t *tmp_out_ptr = (int64_t *)(&vtmp_out);
  int32_t r1 = tmp_out_ptr[0] > max_32bit ? max_32bit : tmp_out_ptr[0];
  r1 = r1 < min_32bit ? min_32bit : r1;
  int32_t r2 = tmp_out_ptr[1] > max_32bit ? max_32bit : tmp_out_ptr[1];
  r2 = r2 < min_32bit ? min_32bit : r2;
  int32_t r3 = tmp_out_ptr[2] > max_32bit ? max_32bit : tmp_out_ptr[2];
  r3 = r3 < min_32bit ? min_32bit : r3;
  int32_t r4 = tmp_out_ptr[3] > max_32bit ? max_32bit : tmp_out_ptr[3];
  r4 = r4 < min_32bit ? min_32bit : r4;
  return _mm_set_epi32(r4, r3, r2, r1);
}
