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

#ifndef MINDSPORE_LITE_NNACL_FP16_EXP_H_
#define MINDSPORE_LITE_NNACL_FP16_EXP_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void ExpFp16(const float16_t *src, float16_t *dst, int num);

#if defined(ENABLE_ARM64)
static inline float32x4_t exp_fp32(float32x4_t input) {
  static float32x4_t param[] = {{0.693147f, 0.693147f, 0.693147f, 0.693147f},
                                {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
                                {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
                                {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
                                {0.5f, 0.5f, 0.5f, 0.5f},
                                {1.0f, 1.0f, 1.0f, 1.0f}};
  int32x4_t integer = vcvtq_s32_f32(input / param[0]);
  float32x4_t decimal = input - vcvtq_f32_s32(integer) * param[0];
  int32x4_t int_exp = vshlq_s32((integer + vmovq_n_s32(127)), vmovq_n_s32(23));
  float32x4_t decimal_exp =
    param[5] +
    decimal * (param[5] + decimal * (param[4] + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  decimal_exp = decimal_exp * vld1q_f32((float *)(&int_exp));
  return decimal_exp;
}

static inline void simd_exp_fp16(float16x8_t input, float16_t *dst) {
  static float16x8_t maxv = {88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f};
  static float16x8_t minv = {-88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f};

  input = vmaxq_f16(minv, vminq_f16(input, maxv));
  float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input));
  float32x4_t input_high = vcvt_high_f32_f16(input);
  vst1q_f16(dst, vcombine_f16(vcvt_f16_f32(exp_fp32(input_low)), vcvt_f16_f32(exp_fp32(input_high))));
}
#endif

static inline void single_exp_fp16(float16_t src, float16_t *dst) {
  static float param[] = {0.693147f, 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};
  src = MSMAX(-88.0f, MSMIN(88.0f, src));
  int integer = (float)src / param[0];
  float decimal = (float)src - integer * param[0];
  int int_exp = (integer + 127) << 23;
  const float decimal_exp =
    1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  *dst = (float16_t)(*((float *)&int_exp) * decimal_exp);
}
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_EXP_H_
