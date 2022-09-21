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

#ifndef MINDSPORE_NNACL_AVX512_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_NNACL_AVX512_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <float.h>
#include <math.h>

#ifdef _MSC_VER
#include <immintrin.h>
#define MS_F32X16_GETI(src, i) src.m512_f32[i]
#define MS512_F32_GETI(src, i) src.m512_f32[i]
#else
#include <x86intrin.h>
#define MS_F32X16_GETI(src, i) src[i]
#define MS512_F32_GETI(src, i) src[i]
#endif

#pragma GCC push_options
#pragma GCC target("avx512f")

#define PI 3.1415926f
#define LN2 0.693147f

#define MS_FLOAT32X16 __m512
#define MS_FLOAT512_F32 __m512
#define MS_INT32X16 __m512i
#define MS_INT512_EPI32 __m512i
#define MS_MASK512_TYPE __mmask16
#define MS_LD512_F32 _mm512_loadu_ps
#define MS_LD512_EPI32(src) _mm512_loadu_si512((__m512i const *)(src))
#define MS_LD512_HALF_EPI32(src) _mm256_loadu_si256((__m256i const *)(src))
#define MS_ADD512_F32 _mm512_add_ps
#define MS_ADD512_EPI32 _mm512_add_epi32
#define MS_MOV512_F32 _mm512_set1_ps
#define MS_MOV512_EPI32 _mm512_set1_epi32
#define MS_MOV512_VAL0_F32 _mm512_setzero_ps()
#define MS_MLA512_F32(src1, src2, src3) _mm512_fmadd_ps(src2, src3, src1)
#define MS_ST512_F32 _mm512_storeu_ps
#define MS_ST512_EPI32(src1, src2) _mm512_storeu_si512((__m512i *)(src1), src2)
#define MS_ST512_HALF_EPI32(src1, src2) _mm256_storeu_si256((__m256i *)(src1), src2)
#define MS_SUB512_F32 _mm512_sub_ps
#define MS_SUB512_EPI32 _mm512_sub_epi32
#define MS_MAX512_F32 _mm512_max_ps
#define MS_MAX512_EPI32 _mm512_max_epi32
#define MS_MIN512_F32 _mm512_min_ps
#define MS_MIN512_EPI32 _mm512_min_epi32
#define MS_SQRT512_F32 _mm512_sqrt_ps
#define MS_RSQRT512_F32 _mm512_rsqrt14_ps
#define MS_SIN512_F32 _mm512_sin_ps
#define MS_ERF512_F32 _mm512_erf_ps
#define MS_ABS512_F32 _mm512_abs_ps
#define MS_ABS512_EPI32 _mm512_abs_epi32

#define MS_ROUND512_F32(src) \
  _mm512_add_round_ps(src, _mm512_set1_ps(0.0f), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define MS_FLOOR512_F32 _mm512_floor_ps
#define MS_CEIL512_F32 _mm512_ceil_ps
#define MS_MUL512_F32(src1, src2) _mm512_mul_ps(src1, src2)
#define MS_MUL512_EPI32(src1, src2) _mm512_mullo_epi32(src1, src2)
#define MS_FMADD512_F32(src1, src2, src3) _mm512_fmadd_ps(src1, src2, src3)
#define MS_FMSUB512_F32(src1, src2, src3) _mm512_fmsub_ps(src1, src2, src3)
#define MS_FSMUL512_F32(src1, src2, src3) _mm512_fnmadd_ps(src3, src2, src1)  // src1 - src2 * src3
#define MS_DIV512_F32(src1, src2) _mm512_div_ps(src1, src2)
#define MS_MUL512_N_F32(src1, src2) _mm512_mul_ps(src1, _mm512_set1_ps(src2))
#define MS_MUL512_N_EPI32(src1, src2) _mm512_mullo_epi32(src1, _mm512_set1_epi32(src2))
#define MS_DIV512_N_F32(src1, src2) _mm512_div_ps(src1, _mm512_set1_ps(src2))
#define MS_SLLI512_EPI32(src1, src2) _mm512_slli_epi32(src1, src2)
#define MS_CVT512PS_EPI32(src) _mm512_cvttps_epi32(src)
#define MS_CVT512EPI32_PS(src) _mm512_cvtepi32_ps(src)  // truncate float to int
#define MS_CMP512_F32(src1, src2, src3) _mm512_cmp_ps_mask(src1, src2, src3)
#define MS_CMPGT512_F32(src1, src2) _mm512_cmp_ps_mask(src1, src2, 30)
#define MS_CMPLE512_F32(src1, src2) _mm512_cmp_ps_mask(src1, src2, 18)
#define MS_CMPLT512_F32(src1, src2) _mm512_cmp_ps_mask(src1, src2, 17)
#define MS_CMPGT512_EPI32(src1, src2) _mm512_cmpgt_epi32(src1, src2)
#define MS_BLEND512_F32(src1, src2, mask) _mm512_mask_blend_ps(mask, src1, src2)
#define MS_BLEND512_EPI32(src1, src2, mask) _mm512_mask_blend_epi32(mask, src1, src2)
#define MS_CAST512_F32_S32(src) _mm512_castsi512_ps(src)
#define MS_REDUCE_ADD512_F32(src) _mm512_reduce_add_ps(src)
#define MS_GET_MAX512_F32(src) _mm512_reduce_max_ps(src)
#define MS_GET_MIN512_F32(src) _mm512_reduce_min_ps(src)
#define MS_GET_SUM512_F32(src) _mm512_reduce_add_ps(src)
#define MS_AND512_MASK(src1, src2) _mm512_kand(src1, src2)

#define MS512_SRLI_EPI32(src1, src2) _mm512_srli_epi32(src1, src2)
#define MS512_AND_EPI32(src1, src2) _mm512_and_si512(src1, src2)
#define MS512_CASTPS_EPI32(src) _mm512_castps_si512(src)
#define MS_OR512_EPI32(src1, src2) _mm512_or_epi32(src1, src2)
#define MS_AND512_EPI32(src1, src2) _mm512_and_epi32(src1, src2)

static inline MS_FLOAT512_F32 MS_OR512_F32(MS_FLOAT512_F32 src1, MS_FLOAT512_F32 src2) {
  MS_FLOAT512_F32 result = MS_CAST512_F32_S32(MS_OR512_EPI32(MS512_CASTPS_EPI32(src1), MS512_CASTPS_EPI32(src2)));
  return result;
}

static inline MS_FLOAT512_F32 MS512_ANDNOT_F32(MS_FLOAT512_F32 src1, MS_FLOAT512_F32 src2) {
  MS_FLOAT512_F32 result = MS_CAST512_F32_S32(MS_AND512_EPI32(~MS512_CASTPS_EPI32(src1), MS512_CASTPS_EPI32(src2)));
  return result;
}

static inline MS_FLOAT512_F32 MS_AND512_MASK_F32(MS_MASK512_TYPE mask, MS_FLOAT512_F32 value) {
  /* mask = T ? value ; 0 */
  MS_FLOAT512_F32 zeros = _mm512_set1_ps(0.0f);
  return _mm512_mask_blend_ps(mask, zeros, value);
}

static inline MS_FLOAT32X16 MS_POW512_F32(MS_FLOAT32X16 src1, MS_FLOAT32X16 src2) {
  MS_FLOAT32X16 dst;
  MS512_F32_GETI(dst, 0) = powf(MS512_F32_GETI(src1, 0), MS512_F32_GETI(src2, 0));
  MS512_F32_GETI(dst, 1) = powf(MS512_F32_GETI(src1, 1), MS512_F32_GETI(src2, 1));
  MS512_F32_GETI(dst, 2) = powf(MS512_F32_GETI(src1, 2), MS512_F32_GETI(src2, 2));
  MS512_F32_GETI(dst, 3) = powf(MS512_F32_GETI(src1, 3), MS512_F32_GETI(src2, 3));
  MS512_F32_GETI(dst, 4) = powf(MS512_F32_GETI(src1, 4), MS512_F32_GETI(src2, 4));
  MS512_F32_GETI(dst, 5) = powf(MS512_F32_GETI(src1, 5), MS512_F32_GETI(src2, 5));
  MS512_F32_GETI(dst, 6) = powf(MS512_F32_GETI(src1, 6), MS512_F32_GETI(src2, 6));
  MS512_F32_GETI(dst, 7) = powf(MS512_F32_GETI(src1, 7), MS512_F32_GETI(src2, 7));
  MS512_F32_GETI(dst, 8) = powf(MS512_F32_GETI(src1, 8), MS512_F32_GETI(src2, 8));
  MS512_F32_GETI(dst, 9) = powf(MS512_F32_GETI(src1, 9), MS512_F32_GETI(src2, 9));
  MS512_F32_GETI(dst, 10) = powf(MS512_F32_GETI(src1, 10), MS512_F32_GETI(src2, 10));
  MS512_F32_GETI(dst, 11) = powf(MS512_F32_GETI(src1, 11), MS512_F32_GETI(src2, 11));
  MS512_F32_GETI(dst, 12) = powf(MS512_F32_GETI(src1, 12), MS512_F32_GETI(src2, 12));
  MS512_F32_GETI(dst, 13) = powf(MS512_F32_GETI(src1, 13), MS512_F32_GETI(src2, 13));
  MS512_F32_GETI(dst, 14) = powf(MS512_F32_GETI(src1, 14), MS512_F32_GETI(src2, 14));
  MS512_F32_GETI(dst, 15) = powf(MS512_F32_GETI(src1, 15), MS512_F32_GETI(src2, 15));
  return dst;
}

static inline MS_FLOAT32X16 MS_COS512_F32(MS_FLOAT32X16 src) {
  static const MS_FLOAT32X16 pi = {PI, PI, PI, PI, PI, PI, PI, PI, PI, PI, PI, PI, PI, PI, PI, PI};
  static const MS_FLOAT32X16 pi2_neg = {-2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI,
                                        -2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI, -2 * PI};
  static const MS_FLOAT32X16 div_pi2 = {1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI),
                                        1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI),
                                        1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI),
                                        1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI), 1.0f / (2 * PI)};
  MS_FLOAT512_F32 src_abs = MS_ABS512_F32(src);
  MS_FLOAT512_F32 src_cycle =
    MS_ADD512_F32(MS_MUL512_F32(MS_FLOOR512_F32(MS_MUL512_F32(MS_ADD512_F32(src_abs, pi), div_pi2)), pi2_neg), src_abs);
  static const MS_FLOAT512_F32 data0 = {1.0f / 90, 1.0f / 90, 1.0f / 90, 1.0f / 90, 1.0f / 90, 1.0f / 90,
                                        1.0f / 90, 1.0f / 90, 1.0f / 90, 1.0f / 90, 1.0f / 90, 1.0f / 90,
                                        1.0f / 90, 1.0f / 90, 1.0f / 90, 1.0f / 90};
  static const MS_FLOAT512_F32 data1 = {1.0f / 56, 1.0f / 56, 1.0f / 56, 1.0f / 56, 1.0f / 56, 1.0f / 56,
                                        1.0f / 56, 1.0f / 56, 1.0f / 56, 1.0f / 56, 1.0f / 56, 1.0f / 56,
                                        1.0f / 56, 1.0f / 56, 1.0f / 56, 1.0f / 56};
  static const MS_FLOAT512_F32 data2 = {1.0f / 30, 1.0f / 30, 1.0f / 30, 1.0f / 30, 1.0f / 30, 1.0f / 30,
                                        1.0f / 30, 1.0f / 30, 1.0f / 30, 1.0f / 30, 1.0f / 30, 1.0f / 30,
                                        1.0f / 30, 1.0f / 30, 1.0f / 30, 1.0f / 30};
  static const MS_FLOAT512_F32 data3 = {1.0f / 12, 1.0f / 12, 1.0f / 12, 1.0f / 12, 1.0f / 12, 1.0f / 12,
                                        1.0f / 12, 1.0f / 12, 1.0f / 12, 1.0f / 12, 1.0f / 12, 1.0f / 12,
                                        1.0f / 12, 1.0f / 12, 1.0f / 12, 1.0f / 12};
  static const MS_FLOAT512_F32 data4 = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                        0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  static const MS_FLOAT32X16 neg = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  static const MS_FLOAT32X16 pos = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  MS_FLOAT32X16 square = MS_MUL512_F32(src_cycle, src_cycle);

  MS_FLOAT32X16 tmp =
    MS_MUL512_F32(MS_MUL512_F32(MS_ADD512_F32(MS_MUL512_F32(MS_MUL512_F32(neg, square), data0), pos), square), data1);
  MS_FLOAT32X16 tmp1 = MS_MUL512_F32(MS_MUL512_F32(MS_ADD512_F32(tmp, neg), square), data2);
  MS_FLOAT512_F32 res = MS_ADD512_F32(
    MS_MUL512_F32(
      MS_MUL512_F32(MS_ADD512_F32(MS_MUL512_F32(MS_MUL512_F32(MS_ADD512_F32(tmp1, pos), square), data3), neg), square),
      data4),
    pos);
  return res;
}

static inline MS_FLOAT32X16 MS512_LOG_F32(MS_FLOAT32X16 src) {
  const MS_INT512_EPI32 gFloatExpMask = MS_MOV512_EPI32(0xffULL << 23);
  const MS_INT512_EPI32 gFloatExp0 = MS_MOV512_EPI32(127ULL << 23);
  const MS_INT512_EPI32 gExpNormalizer = MS_MOV512_EPI32(127);
  static const MS_FLOAT512_F32 data0 = {1.0f / 11, 1.0f / 11, 1.0f / 11, 1.0f / 11, 1.0f / 11, 1.0f / 11,
                                        1.0f / 11, 1.0f / 11, 1.0f / 11, 1.0f / 11, 1.0f / 11, 1.0f / 11,
                                        1.0f / 11, 1.0f / 11, 1.0f / 11, 1.0f / 11};
  static const MS_FLOAT512_F32 data1 = {1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9,
                                        1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9, 1.0f / 9};
  static const MS_FLOAT512_F32 data2 = {1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7,
                                        1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7, 1.0f / 7};
  static const MS_FLOAT512_F32 data3 = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                                        0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
  static const MS_FLOAT512_F32 data4 = {1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
                                        1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3};
  static const MS_FLOAT512_F32 data5 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  static const MS_FLOAT512_F32 data6 = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                                        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  static const MS_FLOAT32X16 neg = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  static const MS_FLOAT32X16 pos = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  static const MS_FLOAT32X16 ln2 = {LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2, LN2};

  const MS_INT512_EPI32 exps32 = MS512_SRLI_EPI32(MS512_AND_EPI32(gFloatExpMask, MS512_CASTPS_EPI32(src)), 23);
  const MS_INT512_EPI32 normExps = MS_SUB512_EPI32(exps32, gExpNormalizer);
  const MS_FLOAT32X16 expsPD = MS_CVT512EPI32_PS(normExps);
  const MS_FLOAT32X16 y =
    MS_OR512_F32(MS_CAST512_F32_S32(gFloatExp0), MS512_ANDNOT_F32(MS_CAST512_F32_S32(gFloatExpMask), src));
  MS_FLOAT32X16 div = MS_DIV512_F32(MS_ADD512_F32(y, neg), MS_ADD512_F32(y, pos));
  MS_FLOAT32X16 square = MS_MUL512_F32(div, div);

  MS_FLOAT32X16 tmp = MS_ADD512_F32(
    MS_MUL512_F32(MS_ADD512_F32(MS_MUL512_F32(square, MS_ADD512_F32(MS_MUL512_F32(square, data0), data1)), data2),
                  square),
    data3);
  MS_FLOAT32X16 tmp1 = MS_MUL512_F32(square, MS_ADD512_F32(MS_MUL512_F32(square, tmp), data4));
  MS_FLOAT32X16 res =
    MS_ADD512_F32(MS_MUL512_F32(ln2, expsPD), MS_MUL512_F32(MS_MUL512_F32(div, MS_ADD512_F32(tmp1, data5)), data6));
  return res;
}

#define MS_DIV512_EPI32(src1, src2) \
  _mm512_cvttps_epi32(MS_DIV512_F32(_mm512_cvtepi32_ps(src1), _mm512_cvtepi32_ps(src2)))

#define MS512_INT16_TO_FLOAT16(src) _mm512_cvtepi16_ph(src)
#define MS512_FLOAT16_TO_INT16(src) _mm512_cvttph_epi16(src)

#define MS512_INT32_TO_FLOAT16(src) _mm512_cvtepi32_ph(src)
#define MS512_FLOAT16_TO_INT32(src) _mm512_cvttph_epi32(src)

#define MS512_INT32_TO_FLOAT32(src) _mm512_cvtepi32_ps(src)
#define MS512_FLOAT32_TO_INT32(src) _mm512_cvttps_epi32(src)
#define MS512_FLOAT16_TO_FLOAT32(src) _mm512_cvtph_ps(src)
#define MS512_FLOAT32_TO_FLOAT16(src1, src2) _mm512_cvtps_ph(src1, src2)

#define MS512_INT64_TO_FLOAT32(src) _mm512_cvtepi64_ps(src)
#define MS512_FLOAT32_TO_INT64(src) _mm512_cvttps_epi64(src)

#define MS512_INT64_TO_FLOAT16(src) _mm512_cvtepi64_ph(src)
#define MS512_FLOAT16_TO_INT64(src) _mm512_cvttph_epi64(src)

#define MS512_INT32_TO_FLOAT64(src) _mm512_cvtepi32_pd(src)
#define MS512_FLOAT64_TO_INT32(src) _mm512_cvttpd_epi32(src)

#define MS512_INT64_TO_FLOAT64(src) _mm512_cvtepi64_pd(src)
#define MS512_FLOAT64_TO_INT64(src) _mm512_cvttpd_epi64(src)

#define MS512_INT16_TO_INT32(src) _mm512_cvtepi16_epi32(src)
#define MS512_INT16_TO_INT64(src) _mm512_cvtepi16_epi64(src)
#define MS512_INT32_TO_INT16(src) _mm512_cvtepi32_epi16(src)
#define MS512_INT32_TO_INT64(src) _mm512_cvtepi32_epi64(src)
#define MS512_INT64_TO_INT16(src) _mm512_cvtepi64_epi16(src)
#define MS512_INT64_TO_INT32(src) _mm512_cvtepi64_epi32(src)

static inline MS_FLOAT32X16 simd_exp512_f32(MS_FLOAT32X16 input) {
  static MS_FLOAT32X16 maxv = {88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f,
                               88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f};
  static MS_FLOAT32X16 minv = {-88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f,
                               -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f};
  static MS_FLOAT32X16 param[] = {
    {0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f,
     0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f},
    {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120,
     1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
    {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24,
     1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
    {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6,
     1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
    {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
  input = MS_MAX512_F32(minv, MS_MIN512_F32(input, maxv));
  MS_INT32X16 integer = MS_CVT512PS_EPI32(MS_DIV512_F32(input, param[0]));
  MS_FLOAT32X16 decimal = MS_SUB512_F32(input, MS_MUL512_F32(MS_CVT512EPI32_PS(integer), param[0]));
  MS_INT32X16 int_exp = MS_SLLI512_EPI32(MS_ADD512_EPI32(integer, MS_MOV512_EPI32(127)), 23);
  MS_FLOAT32X16 tmp = MS_FMADD512_F32(decimal, MS_FMADD512_F32(decimal, param[1], param[2]), param[3]);
  tmp = MS_FMADD512_F32(decimal, MS_FMADD512_F32(decimal, tmp, param[4]), param[5]);
  MS_FLOAT32X16 decimal_exp = MS_FMADD512_F32(decimal, tmp, param[5]);
  return MS_MUL512_F32(decimal_exp, MS_CAST512_F32_S32(int_exp));
}

static inline MS_FLOAT32X16 simd_hexp512_f32(MS_FLOAT32X16 src) {
  MS_FLOAT32X16 dst;
  MS512_F32_GETI(dst, 0) = exp(MS512_F32_GETI(src, 0));
  MS512_F32_GETI(dst, 1) = exp(MS512_F32_GETI(src, 1));
  MS512_F32_GETI(dst, 2) = exp(MS512_F32_GETI(src, 2));
  MS512_F32_GETI(dst, 3) = exp(MS512_F32_GETI(src, 3));
  MS512_F32_GETI(dst, 4) = exp(MS512_F32_GETI(src, 4));
  MS512_F32_GETI(dst, 5) = exp(MS512_F32_GETI(src, 5));
  MS512_F32_GETI(dst, 6) = exp(MS512_F32_GETI(src, 6));
  MS512_F32_GETI(dst, 7) = exp(MS512_F32_GETI(src, 7));
  MS512_F32_GETI(dst, 8) = exp(MS512_F32_GETI(src, 8));
  MS512_F32_GETI(dst, 9) = exp(MS512_F32_GETI(src, 9));
  MS512_F32_GETI(dst, 10) = exp(MS512_F32_GETI(src, 10));
  MS512_F32_GETI(dst, 11) = exp(MS512_F32_GETI(src, 11));
  MS512_F32_GETI(dst, 12) = exp(MS512_F32_GETI(src, 12));
  MS512_F32_GETI(dst, 13) = exp(MS512_F32_GETI(src, 13));
  MS512_F32_GETI(dst, 14) = exp(MS512_F32_GETI(src, 14));
  MS512_F32_GETI(dst, 15) = exp(MS512_F32_GETI(src, 15));
  return dst;
}

static inline void simd_exp512(MS_FLOAT32X16 input, float *dst) {
  MS_FLOAT32X16 res = simd_exp512_f32(input);
  MS_ST512_F32(dst, res);
}

static inline MS_FLOAT32X16 MS_TANHX16_F32(MS_FLOAT32X16 src) {
  static const MS_FLOAT32X16 data0 = {378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f,
                                      378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f};
  static const MS_FLOAT32X16 data1 = {17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f,
                                      17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f};
  static const MS_FLOAT32X16 data2 = {135135.0f, 135135.0f, 135135.0f, 135135.0f, 135135.0f, 135135.0f,
                                      135135.0f, 135135.0f, 135135.0f, 135135.0f, 135135.0f, 135135.0f,
                                      135135.0f, 135135.0f, 135135.0f, 135135.0f};
  static const MS_FLOAT32X16 data3 = {28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f,
                                      28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f};
  static const MS_FLOAT32X16 data4 = {3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f,
                                      3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f};
  static const MS_FLOAT32X16 data5 = {62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f,
                                      62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f};
  static const MS_FLOAT32X16 neg = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  static const MS_FLOAT32X16 pos = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  MS_FLOAT32X16 square = MS_MUL512_F32(src, src);
  MS_FLOAT32X16 a =
    MS_MUL512_F32(MS_FMADD512_F32(MS_FMADD512_F32(MS_ADD512_F32(square, data0), square, data1), square, data2), src);
  MS_FLOAT32X16 b =
    MS_FMADD512_F32(MS_FMADD512_F32(MS_FMADD512_F32(data3, square, data4), square, data5), square, data2);
  return MS_MIN512_F32(MS_MAX512_F32(MS_DIV512_F32(a, b), neg), pos);
}

#define MS_TANH512_F32 MS_TANHX16_F32

static inline MS_FLOAT32X16 MS512_ERF_F32(MS_FLOAT32X16 src) {
  MS_FLOAT32X16 dst;
  MS_F32X16_GETI(dst, 0) = erff(MS_F32X16_GETI(src, 0));
  MS_F32X16_GETI(dst, 1) = erff(MS_F32X16_GETI(src, 1));
  MS_F32X16_GETI(dst, 2) = erff(MS_F32X16_GETI(src, 2));
  MS_F32X16_GETI(dst, 3) = erff(MS_F32X16_GETI(src, 3));
  MS_F32X16_GETI(dst, 4) = erff(MS_F32X16_GETI(src, 4));
  MS_F32X16_GETI(dst, 5) = erff(MS_F32X16_GETI(src, 5));
  MS_F32X16_GETI(dst, 6) = erff(MS_F32X16_GETI(src, 6));
  MS_F32X16_GETI(dst, 7) = erff(MS_F32X16_GETI(src, 7));
  MS_F32X16_GETI(dst, 8) = erff(MS_F32X16_GETI(src, 8));
  MS_F32X16_GETI(dst, 9) = erff(MS_F32X16_GETI(src, 9));
  MS_F32X16_GETI(dst, 10) = erff(MS_F32X16_GETI(src, 10));
  MS_F32X16_GETI(dst, 11) = erff(MS_F32X16_GETI(src, 11));
  MS_F32X16_GETI(dst, 12) = erff(MS_F32X16_GETI(src, 12));
  MS_F32X16_GETI(dst, 13) = erff(MS_F32X16_GETI(src, 13));
  MS_F32X16_GETI(dst, 14) = erff(MS_F32X16_GETI(src, 14));
  MS_F32X16_GETI(dst, 15) = erff(MS_F32X16_GETI(src, 15));
  return dst;
}

#define MS_LOAD512X8_F32(src, input_ptr, num)               \
  MS_FLOAT32X16 src##1 = MS_LD512_F32(input_ptr);           \
  MS_FLOAT32X16 src##2 = MS_LD512_F32(input_ptr + 1 * num); \
  MS_FLOAT32X16 src##3 = MS_LD512_F32(input_ptr + 2 * num); \
  MS_FLOAT32X16 src##4 = MS_LD512_F32(input_ptr + 3 * num); \
  MS_FLOAT32X16 src##5 = MS_LD512_F32(input_ptr + 4 * num); \
  MS_FLOAT32X16 src##6 = MS_LD512_F32(input_ptr + 5 * num); \
  MS_FLOAT32X16 src##7 = MS_LD512_F32(input_ptr + 6 * num); \
  MS_FLOAT32X16 src##8 = MS_LD512_F32(input_ptr + 7 * num);

#define MS_LOAD512X4_F32(src, input_ptr, num)               \
  MS_FLOAT32X16 src##1 = MS_LD512_F32(input_ptr);           \
  MS_FLOAT32X16 src##2 = MS_LD512_F32(input_ptr + 1 * num); \
  MS_FLOAT32X16 src##3 = MS_LD512_F32(input_ptr + 2 * num); \
  MS_FLOAT32X16 src##4 = MS_LD512_F32(input_ptr + 3 * num);

#define MS_FMADD512X8_F32(src, weight, dst)       \
  dst##1 = MS_MLA512_F32(dst##1, src##1, weight); \
  dst##2 = MS_MLA512_F32(dst##2, src##2, weight); \
  dst##3 = MS_MLA512_F32(dst##3, src##3, weight); \
  dst##4 = MS_MLA512_F32(dst##4, src##4, weight); \
  dst##5 = MS_MLA512_F32(dst##5, src##5, weight); \
  dst##6 = MS_MLA512_F32(dst##6, src##6, weight); \
  dst##7 = MS_MLA512_F32(dst##7, src##7, weight); \
  dst##8 = MS_MLA512_F32(dst##8, src##8, weight);

#define MS_FMADD512X4_F32(src, weight, dst)       \
  dst##1 = MS_MLA512_F32(src##1, weight, dst##1); \
  dst##2 = MS_MLA512_F32(src##2, weight, dst##2); \
  dst##3 = MS_MLA512_F32(src##3, weight, dst##3); \
  dst##4 = MS_MLA512_F32(src##4, weight, dst##4);

#define MS_SET_ZERO512X8_F32(dst)             \
  MS_FLOAT32X16 dst##1 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##2 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##3 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##4 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##5 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##6 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##7 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##8 = _mm512_setzero_ps();

#define MS_SET_ZERO512X4_F32(dst)             \
  MS_FLOAT32X16 dst##1 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##2 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##3 = _mm512_setzero_ps(); \
  MS_FLOAT32X16 dst##4 = _mm512_setzero_ps();

#pragma GCC pop_options

#endif  // MINDSPORE_NNACL_AVX512_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
