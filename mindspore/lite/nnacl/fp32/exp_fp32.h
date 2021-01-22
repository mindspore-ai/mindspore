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

#ifndef MINDSPORE_LITE_NNACL_FP32_EXP_H_
#define MINDSPORE_LITE_NNACL_FP32_EXP_H_

#include "nnacl/op_base.h"

typedef struct ExpParameter {
  // Primitive parameter
  OpParameter op_parameter_;
  float base_;
  float scale_;
  float shift_;
  // other parameter
  int thread_num_;
  float in_scale_;
  float out_scale_;
  int element_num_;
} ExpParameter;

#ifdef __cplusplus
extern "C" {
#endif
int Exp(const float *input_data, float *output_data, const ExpParameter *parameter, int task_id);
void ExpFp32(const float *src, float *dst, int num);

#ifdef ENABLE_ARM64
static inline void simd_exp(float32x4_t input4, float *dst) {
  static float32x4_t maxv = {88.0f, 88.0f, 88.0f, 88.0f};
  static float32x4_t minv = {-88.0f, -88.0f, -88.0f, -88.0f};
  static float32x4_t paramv[] = {{0.693147f, 0.693147f, 0.693147f, 0.693147f},
                                 {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
                                 {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
                                 {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
                                 {0.5f, 0.5f, 0.5f, 0.5f},
                                 {1.0f, 1.0f, 1.0f, 1.0f}};
  input4 = vmaxq_f32(minv, vminq_f32(maxv, input4));
  int32x4_t integer4 = vcvtq_s32_f32(vdivq_f32(input4, paramv[0]));
  float32x4_t decimal4 = vsubq_f32(input4, vmulq_f32(vcvtq_f32_s32(integer4), paramv[0]));
  int32x4_t int_exp4 = vshlq_s32(vaddq_s32(integer4, vdupq_n_s32(127)), vdupq_n_s32(23));
  vst1q_f32(dst, vld1q_f32((float32_t *)(&int_exp4)));
  float32x4_t decimal_exp4 = vaddq_f32(paramv[2], vmulq_f32(decimal4, paramv[1]));
  decimal_exp4 = vmulq_f32(decimal4, vaddq_f32(paramv[3], vmulq_f32(decimal4, decimal_exp4)));
  decimal_exp4 = vaddq_f32(paramv[5], vmulq_f32(decimal4, vaddq_f32(paramv[4], decimal_exp4)));
  decimal_exp4 = vaddq_f32(paramv[5], vmulq_f32(decimal4, decimal_exp4));
  vst1q_f32(dst, vmulq_f32(vld1q_f32(dst), decimal_exp4));
}
#endif

static inline void single_exp(float src, float *dst) {
  static float param[] = {0.693147f, 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};  // log(2.0f)
  src = MSMAX(-88.0f, MSMIN(88.0f, src));
  int integer = src / param[0];
  float decimal = src - integer * param[0];
  int int_exp = (integer + 127) << 23;
  memcpy(dst, &int_exp, sizeof(float));
  const float decimal_exp =
    1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  *dst *= decimal_exp;
}
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_EXP_H_
