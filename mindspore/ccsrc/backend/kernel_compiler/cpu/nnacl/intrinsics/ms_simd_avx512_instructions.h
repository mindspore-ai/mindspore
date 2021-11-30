/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version C2NUM.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-C2NUM.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_NNACL_AVX512_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_NNACL_AVX512_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <math.h>

#ifdef _MSC_VER
#include <immintrin.h>
#define MS_F32X16_GETI(src, i) src.m512_f32[i]
#else
#include <x86intrin.h>
#define MS_F32X16_GETI(src, i) src[i]
#endif

#define MS_FLOAT32X16 __m512
#define MS_INT32X16 __m512i
#define MS_LD512_F32 _mm512_loadu_ps
#define MS_LD512_EPI32(src) _mm512_loadu_si512((__m512i const *)(src))
#define MS_ADD512_F32 _mm512_add_ps
#define MS_ADD512_EPI32 _mm512_add_epi32
#define MS_MOV512_F32 _mm512_set1_ps
#define MS_MOV512_EPI32 _mm512_set1_epi32
#define MS_MLA512_F32(src1, src2, src3) _mm512_add_ps(src1, _mm512_mul_ps(src2, src3))
#define MS_ST512_F32 _mm512_storeu_ps
#define MS_ST512_EPI32(src1, src2) _mm512_storeu_si512((__m512i *)(src1), src2)
#define MS_SUB512_F32 _mm512_sub_ps
#define MS_MAX512_F32 _mm512_max_ps
#define MS_MAX512_EPI32 _mm512_max_epi32
#define MS_MIN512_F32 _mm512_min_ps
#define MS_MIN512_EPI32 _mm512_min_epi32
#define MS_MUL512_F32(src1, src2) _mm512_mul_ps(src1, src2)
#define MS_MUL512_EPI32(src1, src2) _mm512_mul_epi32(src1, src2)
#define MS_DIV512_F32(src1, src2) _mm512_div_ps(src1, src2)
#define MS_MUL512_N_F32(src1, src2) _mm512_mul_ps(src1, _mm512_set1_ps(src2))
#define MS_MUL512_N_EPI32(src1, src2) _mm512_mul_epi32(src1, _mm512_set1_epi32(src2))
#define MS_DIV512_N_F32(src1, src2) _mm512_div_ps(src1, _mm512_set1_ps(src2))
#define MS_SLLI512_EPI32(src1, src2) _mm512_slli_epi32(src1, src2)
#define MS_CVT512PS_EPI32(src) _mm512_cvttps_epi32(src)
#define MS_CVT512EPI32_PS(src) _mm512_cvtepi32_ps(src)  // truncate float to int
#define MS_CMP512_F32(src1, src2, src3) _mm512_cmp_ps_mask(src1, src2, src3)
#define MS_CMPGT512_EPI32(src1, src2) _mm512_cmpgt_epi32(src1, src2)
#define MS_BLEND512_F32(src1, src2, src3) _mm512_mask_blend_ps(src1, src2, src3)
#define MS_BLEND512_EPI32(src1, src2, src3) _mm512_mask_blend_epi32(src1, src2, src3)
#define MS_CAST512_F32_S32(src) _mm512_castsi512_ps(src)

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
  MS_FLOAT32X16 a = MS_MUL512_F32(
    MS_ADD512_F32(MS_MUL512_F32(MS_ADD512_F32(MS_MUL512_F32(MS_ADD512_F32(square, data0), square), data1), square),
                  data2),
    src);
  MS_FLOAT32X16 b = MS_ADD512_F32(
    MS_MUL512_F32(MS_ADD512_F32(MS_MUL512_F32(MS_ADD512_F32(MS_MUL512_F32(data3, square), data4), square), data5),
                  square),
    data2);
  return MS_MIN512_F32(MS_MAX512_F32(MS_DIV512_F32(a, b), neg), pos);
}
#endif
