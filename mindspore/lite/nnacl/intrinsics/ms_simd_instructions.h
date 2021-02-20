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

#ifndef MINDSPORE_LITE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#define MINDSPORE_LITE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
#if defined(ENABLE_SSE) || defined(ENABLE_AVX)
#include <x86intrin.h>
#endif

#ifdef ENABLE_ARM
#define MS_FLOAT32X4 float32x4_t
#define MS_INT32X4 int32x4_t
#define MS_LDQ_F32 vld1q_f32
#define MS_LDQ_EPI32 vld1q_s32
#define MS_ADDQ_F32 vaddq_f32
#define MS_ADDQ_EPI32 vaddq_s32
#define MS_MOVQ_F32 vmovq_n_f32
#define MS_MOVQ_EPI32 vmovq_n_s32
#define MS_DUPQ_F32 vdupq_n_f32  // It is recommended to replace with MS_MOVQ_F32.
#define MS_SUBQ_F32 vsubq_f32
#define MS_MLAQ_F32(src1, src2, src3) vmlaq_f32(src1, src2, src3)
#define MS_STQ_F32 vst1q_f32
#define MS_STQ_EPI32 vst1q_s32
#define MS_MAXQ_F32 vmaxq_f32
#define MS_MAXQ_EPI32 vmaxq_s32
#define MS_MINQ_F32 vminq_f32
#define MS_MINQ_EPI32 vminq_s32
#define MS_MULQ_F32(src1, src2) vmulq_f32(src1, src2)
#define MS_MULQ_EPI32(src1, src2) vmulq_s32(src1, src2)
#ifdef ENABLE_ARM64
#define MS_DIVQ_F32(src1, src2) vdivq_f32(src1, src2)
#else
inline static float32x4_t vrecp(float32x4_t v) {
  float32x4_t r = vrecpeq_f32(v);
  r = vmulq_f32(vrecpsq_f32(v, r), r);
  r = vmulq_f32(vrecpsq_f32(v, r), r);
  return r;
}
#define MS_DIVQ_F32(src1, src2) vmulq_f32(src1, vrecp(src2))
#endif
#define MS_MULQ_N_F32(src1, src2) vmulq_n_f32(src1, src2)
#define MS_MULQ_N_EPI32(src1, src2) vmulq_n_s32(src1, src2)
#define MS_DIVQ_N_F32(src1, src2) vdivq_n_f32(src1, src2)
#define MS_SLLIQ_EPI32(src1, src2) vshlq_s32(src1, vmovq_n_s32(src2))
#define MS_CVTQPS_EPI32(src) vcvtq_s32_f32(src)
#define MS_CVTQEPI32_PS(src) vcvtq_f32_s32(src)
#define MS_CMPGTQ_F32(src1, src2) vcgtq_f32(src1, src2)
#define MS_CMPGTQ_EPI32(src1, src2) vcgtq_s32(src1, src2)
// Note: Compared with X86, the vbslq_f32 parameters are the opposite with _mm_blendv_f32
#define MS_BLENDQ_F32(src1, src2, src3) vbslq_f32(src3, src2, src1)
#define MS_BLENDQ_EPI32(src1, src2, src3) vbslq_s32(src3, src2, src1)
#endif

#if defined(ENABLE_AVX)
#define MS_FLOAT32X8 __m256
#define MS_INT32X8 __m256i
#define MS_LD256_F32 _mm256_loadu_ps
#define MS_LD256_EPI32(src) _mm256_loadu_si256((__m256i const *)(src))
#define MS_ADD256_F32 _mm256_add_ps
#define MS_ADD256_EPI32 _mm256_add_epi32
#define MS_MOV256_F32 _mm256_set1_ps
#define MS_MOV256_EPI32 _mm256_set1_epi32
#define MS_DUP256_F32 _mm256_load_ps1  // It is recommended to replace with MS_MOV256_F32.
#define MS_MLA256_F32(src1, src2, src3) _mm256_add_ps(src1, _mm256_mul_ps(src2, src3))
#define MS_ST256_F32 _mm256_storeu_ps
#define MS_ST256_EPI32(src1, src2) _mm256_storeu_si256((__m256i *)(src1), src2)
#define MS_SUB256_F32 _mm256_sub_ps
#define MS_MAX256_F32 _mm256_max_ps
#define MS_MAX256_EPI32 _mm256_max_epi32
#define MS_MIN256_F32 _mm256_min_ps
#define MS_MIN256_EPI32 _mm256_min_epi32
#define MS_MUL256_F32(src1, src2) _mm256_mul_ps(src1, src2)
#define MS_MUL256_EPI32(src1, src2) _mm256_mul_epi32(src1, src2)
#define MS_DIV256_F32(src1, src2) _mm256_div_ps(src1, src2)
#define MS_MUL256_N_F32(src1, src2) _mm256_mul_ps(src1, _mm256_set1_ps(src2))
#define MS_MUL256_N_EPI32(src1, src2) _mm256_mul_epi32(src1, _mm256_set1_epi32(src2))
#define MS_DIV256_N_F32(src1, src2) _mm256_div_ps(src1, _mm256_set1_ps(src2))
#define MS_SLLI256_EPI32(src1, src2) _mm256_slli_epi32(src1, src2)
#define MS_CVT256PS_EPI32(src) _mm256_cvttps_epi32(src)
#define MS_CVT256EPI32_PS(src) _mm256_cvtepi32_ps(src)  // truncate float to int
#define MS_CMP256_F32(src1, src2, src3) _mm256_cmp_ps(src1, src2, src3)
#define MS_CMPGT256_EPI32(src1, src2) _mm256_cmpgt_epi32(src1, src2)
#define MS_BLEND256_F32(src1, src2, src3) _mm256_blendv_ps(src1, src2, src3)
#define MS_BLEND256_EPI32(src1, src2, src3) _mm256_blendv_epi8(src1, src2, src3)
#endif

#if defined(ENABLE_SSE)
#define MS_FLOAT32X4 __m128
#define MS_INT32X4 __m128i
#define MS_LDQ_F32 _mm_loadu_ps
#define MS_LDQ_EPI32(src) _mm_loadu_si128((__m128i const *)(src))
#define MS_ADDQ_F32 _mm_add_ps
#define MS_ADDQ_EPI32 _mm_add_epi32
#define MS_MOVQ_F32 _mm_set1_ps
#define MS_MOVQ_EPI32 _mm_set1_epi32
#define MS_DUPQ_F32 _mm_load_ps1  // It is recommended to replace with MS_MOVQ_F32.
#define MS_MLAQ_F32(src1, src2, src3) _mm_add_ps(src1, _mm_mul_ps(src2, src3))
#define MS_STQ_F32 _mm_storeu_ps
#define MS_STQ_EPI32(src1, src2) _mm_storeu_si128((__m128i *)(src1), src2)
#define MS_SUBQ_F32 _mm_sub_ps
#define MS_MAXQ_F32 _mm_max_ps
#define MS_MAXQ_EPI32 _mm_max_epi32
#define MS_MINQ_F32 _mm_min_ps
#define MS_MINQ_EPI32 _mm_min_epi32
#define MS_MULQ_F32(src1, src2) _mm_mul_ps(src1, src2)
#define MS_MULQ_EPI32(src1, src2) _mm_mul_epi32(src1, src2)
#define MS_DIVQ_F32(src1, src2) _mm_div_ps(src1, src2)
#define MS_MULQ_N_F32(src1, src2) _mm_mul_ps(src1, _mm_set1_ps(src2))
#define MS_MULQ_N_EPI32(src1, src2) _mm_mul_epi32(src1, _mm_set1_epi32(src2))
#define MS_DIVQ_N_F32(src1, src2) _mm_div_ps(src1, _mm_set1_ps(src2))
#define MS_SLLIQ_EPI32(src1, src2) _mm_slli_epi32(src1, src2)
#define MS_CVTQPS_EPI32(src) _mm_cvttps_epi32(src)  // truncate float to int
#define MS_CVTQEPI32_PS(src) _mm_cvtepi32_ps(src)
#define MS_CMPGTQ_F32(src1, src2) _mm_cmpgt_ps(src1, src2)
#define MS_CMPGTQ_EPI32(src1, src2) _mm_cmpgt_epi32(src1, src2)
#define MS_BLENDQ_F32(src1, src2, src3) _mm_blendv_ps(src1, src2, src3)
#define MS_BLENDQ_EPI32(src1, src2, src3) _mm_blendv_epi8(src1, src2, src3)
#endif

#define LOAD256X8_F32(src, input_ptr, num)                 \
  MS_FLOAT32X8 src##1 = MS_LD256_F32(input_ptr + 0 * num); \
  MS_FLOAT32X8 src##2 = MS_LD256_F32(input_ptr + 1 * num); \
  MS_FLOAT32X8 src##3 = MS_LD256_F32(input_ptr + 2 * num); \
  MS_FLOAT32X8 src##4 = MS_LD256_F32(input_ptr + 3 * num); \
  MS_FLOAT32X8 src##5 = MS_LD256_F32(input_ptr + 4 * num); \
  MS_FLOAT32X8 src##6 = MS_LD256_F32(input_ptr + 5 * num); \
  MS_FLOAT32X8 src##7 = MS_LD256_F32(input_ptr + 6 * num); \
  MS_FLOAT32X8 src##8 = MS_LD256_F32(input_ptr + 7 * num);

#define STORE256X8_F32(output_ptr, num, dst)  \
  MS_ST256_F32(output_ptr + 0 * num, dst##1); \
  MS_ST256_F32(output_ptr + 1 * num, dst##2); \
  MS_ST256_F32(output_ptr + 2 * num, dst##3); \
  MS_ST256_F32(output_ptr + 3 * num, dst##4); \
  MS_ST256_F32(output_ptr + 4 * num, dst##5); \
  MS_ST256_F32(output_ptr + 5 * num, dst##6); \
  MS_ST256_F32(output_ptr + 6 * num, dst##7); \
  MS_ST256_F32(output_ptr + 7 * num, dst##8);

#define LOAD128X8_F32(src, input_ptr, num)               \
  MS_FLOAT32X4 src##1 = MS_LDQ_F32(input_ptr + 0 * num); \
  MS_FLOAT32X4 src##2 = MS_LDQ_F32(input_ptr + 1 * num); \
  MS_FLOAT32X4 src##3 = MS_LDQ_F32(input_ptr + 2 * num); \
  MS_FLOAT32X4 src##4 = MS_LDQ_F32(input_ptr + 3 * num); \
  MS_FLOAT32X4 src##5 = MS_LDQ_F32(input_ptr + 4 * num); \
  MS_FLOAT32X4 src##6 = MS_LDQ_F32(input_ptr + 5 * num); \
  MS_FLOAT32X4 src##7 = MS_LDQ_F32(input_ptr + 6 * num); \
  MS_FLOAT32X4 src##8 = MS_LDQ_F32(input_ptr + 7 * num);

#define STORE128X8_F32(output_ptr, num, dst) \
  MS_STQ_F32(output_ptr + 0 * num, dst##1);  \
  MS_STQ_F32(output_ptr + 1 * num, dst##2);  \
  MS_STQ_F32(output_ptr + 2 * num, dst##3);  \
  MS_STQ_F32(output_ptr + 3 * num, dst##4);  \
  MS_STQ_F32(output_ptr + 4 * num, dst##5);  \
  MS_STQ_F32(output_ptr + 5 * num, dst##6);  \
  MS_STQ_F32(output_ptr + 6 * num, dst##7);  \
  MS_STQ_F32(output_ptr + 7 * num, dst##8);

#endif  // MINDSPORE_LITE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_H_
