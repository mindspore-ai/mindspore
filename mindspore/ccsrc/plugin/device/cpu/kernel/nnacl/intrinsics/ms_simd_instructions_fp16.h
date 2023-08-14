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
#ifndef NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_FP16_H_
#define NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_FP16_H_
#include <math.h>
#include "nnacl/intrinsics/ms_simd_instructions.h"

#if defined(ENABLE_ARM82_A32)
static inline float16x8_t ms_vdivq_f16(float16x8_t in1, float16x8_t in2) {
  float16x8_t dst;
  asm volatile(
    "vrecpe.f16 q14, %3\n"
    "vrecps.f16 q15, %3, q14\n"
    "vmul.f16 q14, q15, q14\n"
    "vrecps.f16 q15, %3, q14\n"
    "vmul.f16 q14, q15, q14\n"
    "vmul.f16 %0, %2, q14\n"
    : "=w"(dst)
    : "0"(dst), "w"(in1), "w"(in2)
    : "q14", "q15");
  return dst;
}

static inline float16x4_t ms_vdiv_f16(float16x4_t in1, float16x4_t in2) {
  float16x4_t dst;
  asm volatile(
    "vrecpe.f16 d14, %3\n"
    "vrecps.f16 d16, %3, d14\n"
    "vmul.f16 d14, d16, d14\n"
    "vrecps.f16 d16, %3, d14\n"
    "vmul.f16 d14, d16, d14\n"
    "vmul.f16 %0, %2, d14\n"
    : "=w"(dst)
    : "0"(dst), "w"(in1), "w"(in2)
    : "d14", "d16");
  return dst;
}

static inline float ms_vaddvq_f32(float32x4_t in) {
  // is not support in arm82 aarch32 and there is no assembly instruction to process all the data
  return in[0] + in[1] + in[2] + in[3];
}

static inline float16_t ms_vmaxvq_f16(float16x8_t in) {
  // is not support in arm82 aarch32 and there is no assembly instruction to process all the data
  float16_t dst = in[0];
  for (int i = 1; i < 8; ++i) {
    dst = dst > in[i] ? dst : in[i];
  }
  return dst;
}

static inline float32x4_t ms_vcvt_f32_f16(float16x4_t in) {
  float32x4_t dst;
  asm volatile("vcvt.f32.f16 %0, %2\n" : "=w"(dst) : "0"(dst), "w"(in) :);
  return dst;
}

static inline float16x4_t ms_vcvt_f16_f32(float32x4_t in) {
  float16x4_t dst;
  asm volatile("vcvt.f16.f32 %0, %2\n" : "=w"(dst) : "0"(dst), "w"(in) :);
  return dst;
}

#define MS_CVT_F32_F16(src) ms_vcvt_f32_f16(src)
#define MS_CVT_F16_F32(src) ms_vcvt_f16_f32(src)
#define MS_DIV_F16(src1, src2) ms_vdiv_f16(src1, src2)
#define MS_DIVQ_F16(src1, src2) ms_vdivq_f16(src1, src2)
#define MS_FMAQ_N_F16(src1, src2, src3) vfmaq_f16(src1, src2, vdupq_n_f16(src3))
#define MS_MAXVQ_F16(src) ms_vmaxvq_f16(src)
#define MS_ADDVQ_F32(src) ms_vaddvq_f32(src)
#else
#define MS_CVT_F32_F16(src) vcvt_f32_f16(src)
#define MS_CVT_F16_F32(src) vcvt_f16_f32(src)
#define MS_DIV_F16(src1, src2) vdiv_f16(src1, src2)
#define MS_DIVQ_F16(src1, src2) vdivq_f16(src1, src2)
#define MS_FMAQ_N_F16(src1, src2, src3) vfmaq_n_f16(src1, src2, src3)
#define MS_MAXVQ_F16(src) vmaxvq_f16(src)
#define MS_ADDVQ_F32(src) vaddvq_f32(src)
#endif

#define MS_FLOAT16X8 float16x8_t
#define MS_FLOAT16X4 float16x4_t
#define MS_FLOAT16X4X4 float16x4x4_t
#define MS_FLOAT16X4X2 float16x4x2_t
#define MS_MOVQ_F16 vmovq_n_f16
#define MS_STQ_F16(ptr, val) vst1q_f16(ptr, val)
#define MS_ST_F16 vst1_f16
#define MS_ST2_F16 vst2_f16
#define MS_ST4_F16 vst4_f16
#define MS_MINQ_F16 vminq_f16
#define MS_MAXQ_F16 vmaxq_f16
#define MS_LDQ_F16(ptr) vld1q_f16(ptr)
#define MS_LD_F16(ptr) vld1_f16(ptr)
#define MS_ADDQ_F16 vaddq_f16
#define MS_SUBQ_F16 vsubq_f16
#define MS_MULQ_F16 vmulq_f16
#define MS_FMAQ_F16 vfmaq_f16
#define MS_MULQ_N_F16(vector, scalar) vmulq_n_f16(vector, scalar)
#define MS_CMPGTQ_F16(src1, src2) vcgtq_f16(src1, src2)

static inline float16x8_t MS_TANHX8_F16(float16x8_t src) {
  float32x4_t src_low = MS_CVT_F32_F16(vget_low_f16(src));
  float32x4_t src_high = MS_CVT_F32_F16(vget_high_f16(src));
  return vcombine_f16(MS_CVT_F16_F32(MS_TANHX4_F32(src_low)), MS_CVT_F16_F32(MS_TANHX4_F32(src_high)));
}

static inline float16x8_t MS_ERFX8_F16(float16x8_t src) {
  float16x8_t dst;
  dst[0] = erff(src[0]);
  dst[1] = erff(src[1]);
  dst[2] = erff(src[2]);
  dst[3] = erff(src[3]);
  dst[4] = erff(src[4]);
  dst[5] = erff(src[5]);
  dst[6] = erff(src[6]);
  dst[7] = erff(src[7]);
  return dst;
}

static inline float16x8_t MS_SQRTFX8_F16(float16x8_t src) {
  float16x8_t dst;
  dst[0] = sqrtf(src[0]);
  dst[1] = sqrtf(src[1]);
  dst[2] = sqrtf(src[2]);
  dst[3] = sqrtf(src[3]);
  dst[4] = sqrtf(src[4]);
  dst[5] = sqrtf(src[5]);
  dst[6] = sqrtf(src[6]);
  dst[7] = sqrtf(src[7]);
  return dst;
}

static inline float16x4_t MS_SQRTFX4_F16(float16x4_t src) {
  float16x4_t dst;
  dst[0] = sqrtf(src[0]);
  dst[1] = sqrtf(src[1]);
  dst[2] = sqrtf(src[2]);
  dst[3] = sqrtf(src[3]);
  return dst;
}

static inline float32x4_t MS_VMLAL_F16(float16x4_t x, float16x4_t dy, float32x4_t sum) {
  float32x4_t x_fp32 = MS_CVT_F32_F16(x);
  float32x4_t dy_fp32 = MS_CVT_F32_F16(dy);
  return vmlaq_f32(sum, x_fp32, dy_fp32);
}

#endif  // NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_FP16_H_
