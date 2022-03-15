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

#ifndef MINDSPORE_NNACL_AVX_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_NNACL_AVX_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#include <math.h>

#ifdef _MSC_VER
#include <immintrin.h>
#define MS_F32X8_GETI(src, i) src.m256_f32[i]
#else
#include <x86intrin.h>
#define MS_F32X8_GETI(src, i) src[i]
#endif

#define MS_FLOAT32X8 __m256
#define MS_INT32X8 __m256i
#define MS_MASK256_TYPE MS_FLOAT32X8
#define MS_LD256_F32 _mm256_loadu_ps
#define MS_LD256_EPI32(src) _mm256_loadu_si256((__m256i const *)(src))
#define MS_ADD256_F32 _mm256_add_ps
#define MS_ADD256_EPI32 _mm256_add_epi32
#define MS_MOV256_F32 _mm256_set1_ps
#define MS_MOV256_EPI32 _mm256_set1_epi32
#define MS_MLA256_F32(src1, src2, src3) _mm256_fmadd_ps(src2, src3, src1)
#define MS_ST256_F32 _mm256_storeu_ps
#define MS_ST256_EPI32(src1, src2) _mm256_storeu_si256((__m256i *)(src1), src2)
#define MS_SUB256_F32 _mm256_sub_ps
#define MS_SUB256_EPI32 _mm256_sub_epi32
#define MS_MAX256_F32 _mm256_max_ps
#define MS_MAX256_EPI32 _mm256_max_epi32
#define MS_MIN256_F32 _mm256_min_ps
#define MS_MIN256_EPI32 _mm256_min_epi32
#define MS_SQRT256_F32 _mm256_sqrt_ps
#define MS_RSQRT256_F32 _mm256_rsqrt_ps
#define MS_LOG256_F32 _mm256_log_ps
#define MS_COS256_F32 _mm256_cos_ps
#define MS_SIN256_F32 _mm256_sin_ps
#define MS_ERF256_F32 _mm256_erf_ps
#define MS_ROUND256_F32(src) _mm256_round_ps(src, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define MS_FLOOR256_F32 _mm256_floor_ps
#define MS_CEIL256_F32 _mm256_ceil_ps
#define MS_MUL256_F32(src1, src2) _mm256_mul_ps(src1, src2)
#define MS_MUL256_EPI32(src1, src2) _mm256_mullo_epi32(src1, src2)
#define MS_DIV256_F32(src1, src2) _mm256_div_ps(src1, src2)
#define MS_MUL256_N_F32(src1, src2) _mm256_mul_ps(src1, _mm256_set1_ps(src2))
#define MS_MUL256_N_EPI32(src1, src2) _mm256_mullo_epi32(src1, _mm256_set1_epi32(src2))
#define MS_DIV256_N_F32(src1, src2) _mm256_div_ps(src1, _mm256_set1_ps(src2))
#define MS_SLLI256_EPI32(src1, src2) _mm256_slli_epi32(src1, src2)
#define MS_CVT256PS_EPI32(src) _mm256_cvttps_epi32(src)
#define MS_CVT256EPI32_PS(src) _mm256_cvtepi32_ps(src)  // truncate float to int
#define MS_CMP256_F32(src1, src2, src3) _mm256_cmp_ps(src1, src2, src3)
#define MS_CMPGT256_F32(src1, src2) _mm256_cmp_ps(src1, src2, 30)
#define MS_CMPLE256_F32(src1, src2) _mm256_cmp_ps(src1, src2, 18)
#define MS_CMPGT256_EPI32(src1, src2) _mm256_cmpgt_epi32(src1, src2)
#define MS_BLEND256_F32(src1, src2, src3) _mm256_blendv_ps(src1, src2, src3)
#define MS_BLEND256_EPI32(src1, src2, src3) _mm256_blendv_epi8(src1, src2, src3)
#define MS_CAST256_F32_S32(src) _mm256_castsi256_ps(src)

static inline float MS_GET_MAX256_F32(__m256 src) {
  float result = MS_F32X8_GETI(src, 0);
  for (int i = 1; i < 8; i++) {  // avx block num : 8
    result = fmaxf(result, MS_F32X8_GETI(src, i));
  }
  return result;
}

#define MS_DIV256_EPI32(src1, src2) \
  _mm256_cvttps_epi32(MS_DIV256_F32(_mm256_cvtepi32_ps(src1), _mm256_cvtepi32_ps(src2)))

static inline MS_FLOAT32X8 MS_SQRTFX8_F32(MS_FLOAT32X8 src) {
  MS_FLOAT32X8 dst;
  MS_F32X8_GETI(dst, 0) = sqrtf(MS_F32X8_GETI(src, 0));
  MS_F32X8_GETI(dst, 1) = sqrtf(MS_F32X8_GETI(src, 1));
  MS_F32X8_GETI(dst, 2) = sqrtf(MS_F32X8_GETI(src, 2));
  MS_F32X8_GETI(dst, 3) = sqrtf(MS_F32X8_GETI(src, 3));
  MS_F32X8_GETI(dst, 4) = sqrtf(MS_F32X8_GETI(src, 4));
  MS_F32X8_GETI(dst, 5) = sqrtf(MS_F32X8_GETI(src, 5));
  MS_F32X8_GETI(dst, 6) = sqrtf(MS_F32X8_GETI(src, 6));
  MS_F32X8_GETI(dst, 7) = sqrtf(MS_F32X8_GETI(src, 7));
  return dst;
}

#define MS_LOAD256X8_F32(src, input_ptr, num)              \
  MS_FLOAT32X8 src##1 = MS_LD256_F32(input_ptr + 0 * num); \
  MS_FLOAT32X8 src##2 = MS_LD256_F32(input_ptr + 1 * num); \
  MS_FLOAT32X8 src##3 = MS_LD256_F32(input_ptr + 2 * num); \
  MS_FLOAT32X8 src##4 = MS_LD256_F32(input_ptr + 3 * num); \
  MS_FLOAT32X8 src##5 = MS_LD256_F32(input_ptr + 4 * num); \
  MS_FLOAT32X8 src##6 = MS_LD256_F32(input_ptr + 5 * num); \
  MS_FLOAT32X8 src##7 = MS_LD256_F32(input_ptr + 6 * num); \
  MS_FLOAT32X8 src##8 = MS_LD256_F32(input_ptr + 7 * num);

#define MS_LOAD256X16_F32(src, input_ptr, num)               \
  MS_FLOAT32X8 src##1 = MS_LD256_F32(input_ptr + 0 * num);   \
  MS_FLOAT32X8 src##2 = MS_LD256_F32(input_ptr + 1 * num);   \
  MS_FLOAT32X8 src##3 = MS_LD256_F32(input_ptr + 2 * num);   \
  MS_FLOAT32X8 src##4 = MS_LD256_F32(input_ptr + 3 * num);   \
  MS_FLOAT32X8 src##5 = MS_LD256_F32(input_ptr + 4 * num);   \
  MS_FLOAT32X8 src##6 = MS_LD256_F32(input_ptr + 5 * num);   \
  MS_FLOAT32X8 src##7 = MS_LD256_F32(input_ptr + 6 * num);   \
  MS_FLOAT32X8 src##8 = MS_LD256_F32(input_ptr + 7 * num);   \
  MS_FLOAT32X8 src##9 = MS_LD256_F32(input_ptr + 8 * num);   \
  MS_FLOAT32X8 src##10 = MS_LD256_F32(input_ptr + 9 * num);  \
  MS_FLOAT32X8 src##11 = MS_LD256_F32(input_ptr + 10 * num); \
  MS_FLOAT32X8 src##12 = MS_LD256_F32(input_ptr + 11 * num); \
  MS_FLOAT32X8 src##13 = MS_LD256_F32(input_ptr + 12 * num); \
  MS_FLOAT32X8 src##14 = MS_LD256_F32(input_ptr + 13 * num); \
  MS_FLOAT32X8 src##15 = MS_LD256_F32(input_ptr + 14 * num); \
  MS_FLOAT32X8 src##16 = MS_LD256_F32(input_ptr + 15 * num);

#define STORE256X8_F32(output_ptr, num, dst)  \
  MS_ST256_F32(output_ptr + 0 * num, dst##1); \
  MS_ST256_F32(output_ptr + 1 * num, dst##2); \
  MS_ST256_F32(output_ptr + 2 * num, dst##3); \
  MS_ST256_F32(output_ptr + 3 * num, dst##4); \
  MS_ST256_F32(output_ptr + 4 * num, dst##5); \
  MS_ST256_F32(output_ptr + 5 * num, dst##6); \
  MS_ST256_F32(output_ptr + 6 * num, dst##7); \
  MS_ST256_F32(output_ptr + 7 * num, dst##8);

#define STORE256X16_F32(output_ptr, num, dst)   \
  MS_ST256_F32(output_ptr + 0 * num, dst##1);   \
  MS_ST256_F32(output_ptr + 1 * num, dst##2);   \
  MS_ST256_F32(output_ptr + 2 * num, dst##3);   \
  MS_ST256_F32(output_ptr + 3 * num, dst##4);   \
  MS_ST256_F32(output_ptr + 4 * num, dst##5);   \
  MS_ST256_F32(output_ptr + 5 * num, dst##6);   \
  MS_ST256_F32(output_ptr + 6 * num, dst##7);   \
  MS_ST256_F32(output_ptr + 7 * num, dst##8);   \
  MS_ST256_F32(output_ptr + 8 * num, dst##9);   \
  MS_ST256_F32(output_ptr + 9 * num, dst##10);  \
  MS_ST256_F32(output_ptr + 10 * num, dst##11); \
  MS_ST256_F32(output_ptr + 11 * num, dst##12); \
  MS_ST256_F32(output_ptr + 12 * num, dst##13); \
  MS_ST256_F32(output_ptr + 13 * num, dst##14); \
  MS_ST256_F32(output_ptr + 14 * num, dst##15); \
  MS_ST256_F32(output_ptr + 15 * num, dst##16);

static inline MS_FLOAT32X8 MS_TANHX8_F32(MS_FLOAT32X8 src) {
  static const MS_FLOAT32X8 data0 = {378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f, 378.0f};
  static const MS_FLOAT32X8 data1 = {17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f, 17325.0f};
  static const MS_FLOAT32X8 data2 = {135135.0f, 135135.0f, 135135.0f, 135135.0f,
                                     135135.0f, 135135.0f, 135135.0f, 135135.0f};
  static const MS_FLOAT32X8 data3 = {28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f, 28.0f};
  static const MS_FLOAT32X8 data4 = {3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f, 3150.0f};
  static const MS_FLOAT32X8 data5 = {62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f, 62370.0f};
  static const MS_FLOAT32X8 neg = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
  static const MS_FLOAT32X8 pos = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  MS_FLOAT32X8 square = MS_MUL256_F32(src, src);
  MS_FLOAT32X8 a = MS_MUL256_F32(
    MS_ADD256_F32(MS_MUL256_F32(MS_ADD256_F32(MS_MUL256_F32(MS_ADD256_F32(square, data0), square), data1), square),
                  data2),
    src);
  MS_FLOAT32X8 b = MS_ADD256_F32(
    MS_MUL256_F32(MS_ADD256_F32(MS_MUL256_F32(MS_ADD256_F32(MS_MUL256_F32(data3, square), data4), square), data5),
                  square),
    data2);
  return MS_MIN256_F32(MS_MAX256_F32(MS_DIV256_F32(a, b), neg), pos);
}

#define MS_FMADD256X8_F32(src, weight, dst)       \
  dst##1 = MS_MLA256_F32(dst##1, src##1, weight); \
  dst##2 = MS_MLA256_F32(dst##2, src##2, weight); \
  dst##3 = MS_MLA256_F32(dst##3, src##3, weight); \
  dst##4 = MS_MLA256_F32(dst##4, src##4, weight); \
  dst##5 = MS_MLA256_F32(dst##5, src##5, weight); \
  dst##6 = MS_MLA256_F32(dst##6, src##6, weight); \
  dst##7 = MS_MLA256_F32(dst##7, src##7, weight); \
  dst##8 = MS_MLA256_F32(dst##8, src##8, weight);

#define MS_SET_ZERO256X8_F32(dst)            \
  MS_FLOAT32X8 dst##1 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##2 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##3 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##4 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##5 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##6 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##7 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##8 = _mm256_setzero_ps();

#define MS_FMADD256X4_F32(src, weight, dst)       \
  dst##1 = MS_MLA256_F32(dst##1, src##1, weight); \
  dst##2 = MS_MLA256_F32(dst##2, src##2, weight); \
  dst##3 = MS_MLA256_F32(dst##3, src##3, weight); \
  dst##4 = MS_MLA256_F32(dst##4, src##4, weight);

#define MS_SET_ZERO256X4_F32(dst)            \
  MS_FLOAT32X8 dst##1 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##2 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##3 = _mm256_setzero_ps(); \
  MS_FLOAT32X8 dst##4 = _mm256_setzero_ps();
#endif  // MINDSPORE_NNACL_AVX_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
