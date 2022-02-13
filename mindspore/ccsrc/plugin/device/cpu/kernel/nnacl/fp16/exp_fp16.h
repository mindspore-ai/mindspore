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
#ifndef MINDSPORE_NNACL_FP16_EXP_FP16_H_
#define MINDSPORE_NNACL_FP16_EXP_FP16_H_

#include "nnacl/op_base.h"
#include "nnacl/exp_parameter.h"
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
void ExpFp16(const float16_t *src, float16_t *dst, int num);
int ExpFusionFp16(const float16_t *src, float16_t *dst, const ExpParameter *param, int task_id);

#ifdef ENABLE_NEON
static inline float16x8_t VexpFp16(float16x8_t input) {
  float32x4_t input_low = MS_CVT_F32_F16(vget_low_f16(input));
  float32x4_t input_high = MS_CVT_F32_F16(vget_high_f16(input));
  return vcombine_f16(MS_CVT_F16_F32(VexpFp32(input_low)), MS_CVT_F16_F32(VexpFp32(input_high)));
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
  float *tmp = (float *)(&int_exp);
  *dst = (float16_t)(*(tmp)*decimal_exp);
}

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_EXP_FP16_H_
