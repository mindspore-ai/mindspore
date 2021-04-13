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
#ifndef MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_FP16_H_
#define MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_FP16_H_
#include <math.h>
#include "nnacl/intrinsics/ms_simd_instructions.h"

#if defined(ENABLE_ARM82_A32)
static inline float16x8_t divq_f16(float16x8_t in1, float16x8_t in2) {
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

static inline float16x4_t div_f16(float16x4_t in1, float16x4_t in2) {
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

static inline float vaddvq_f32(float32x4_t in) {  // is not support in arm82 aarch32
  return in[0] + in[1] + in[2] + in[3];
}

static inline float32x4_t cvt_f32_f16(float16x4_t in) {
  float32x4_t dst;
  asm volatile("vcvt.f32.f16 %0, %2\n" : "=w"(dst) : "0"(dst), "w"(in) :);
  return dst;
}

static inline float16x4_t cvt_f16_f32(float32x4_t in) {
  float16x4_t dst;
  asm volatile("vcvt.f16.f32 %0, %2\n" : "=w"(dst) : "0"(dst), "w"(in) :);
  return dst;
}

#define MS_CVT_F32_F16(src) cvt_f32_f16(src)
#define MS_CVT_F16_F32(src) cvt_f16_f32(src)
#define MS_DIV_F16(src1, src2) div_f16(src1, src2)
#define MS_DIVQ_F16(src1, src2) divq_f16(src1, src2)
#define MS_FMAQ_N_F16(src1, src2, src3) vfmaq_f16(src1, src2, vdupq_n_f16(src3))
#else
#define MS_CVT_F32_F16(src) vcvt_f32_f16(src)
#define MS_CVT_F16_F32(src) vcvt_f16_f32(src)
#define MS_DIV_F16(src1, src2) vdiv_f16(src1, src2)
#define MS_DIVQ_F16(src1, src2) vdivq_f16(src1, src2)
#define MS_FMAQ_N_F16(src1, src2, src3) vfmaq_n_f16(src1, src2, src3)
#endif

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
#endif  // MINDSPORE_NNACL_INTRINSICS_MS_SIMD_INSTRUCTIONS_FP16_H_
