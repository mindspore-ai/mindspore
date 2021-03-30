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

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
static inline void simd_exp(MS_FLOAT32X4 input, float *dst) {
  static MS_FLOAT32X4 maxv = {88.0f, 88.0f, 88.0f, 88.0f};
  static MS_FLOAT32X4 minv = {-88.0f, -88.0f, -88.0f, -88.0f};
  static MS_FLOAT32X4 param[] = {{0.693147f, 0.693147f, 0.693147f, 0.693147f},
                                 {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
                                 {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
                                 {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
                                 {0.5f, 0.5f, 0.5f, 0.5f},
                                 {1.0f, 1.0f, 1.0f, 1.0f}};

  input = MS_MAXQ_F32(minv, MS_MINQ_F32(input, maxv));
  MS_INT32X4 integer = MS_CVTQPS_EPI32(input / param[0]);
  MS_FLOAT32X4 decimal = input - MS_CVTQEPI32_PS(integer) * param[0];
  MS_INT32X4 int_exp = MS_SLLIQ_EPI32(MS_ADDQ_EPI32(integer, MS_MOVQ_EPI32(127)), 23);
  MS_FLOAT32X4 decimal_exp =
    param[5] +
    decimal * (param[5] + decimal * (param[4] + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  MS_STQ_F32(dst, decimal_exp * MS_CAST_F32_S32(int_exp));
}
#endif

#if defined(ENABLE_AVX)
static inline void simd_exp_avx(MS_FLOAT32X8 input, float *dst) {
  static MS_FLOAT32X8 maxv = {88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f};
  static MS_FLOAT32X8 minv = {-88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f};
  static MS_FLOAT32X8 param[] = {
    {0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f, 0.693147f},
    {1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120, 1.0f / 120},
    {1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24, 1.0f / 24},
    {1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6, 1.0f / 6},
    {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
  input = MS_MAX256_F32(minv, MS_MIN256_F32(input, maxv));
  MS_INT32X8 integer = MS_CVT256PS_EPI32(input / param[0]);
  MS_FLOAT32X8 decimal = input - MS_CVT256EPI32_PS(integer) * param[0];
  MS_INT32X8 int_exp = MS_SLLI256_EPI32(MS_ADD256_EPI32(integer, MS_MOV256_EPI32(127)), 23);
  MS_FLOAT32X8 decimal_exp =
    param[5] +
    decimal * (param[5] + decimal * (param[4] + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  MS_ST256_F32(dst, decimal_exp * MS_CAST256_F32_S32(int_exp));
}
#endif

static inline void single_exp(float src, float *dst) {
  typedef union {
    float f;
    int i;
  } fi;
  static float param[] = {0.693147f, 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};  // log(2.0f)
  src = MSMAX(-88.0f, MSMIN(88.0f, src));
  int integer = src / param[0];
  float decimal = src - integer * param[0];
  fi int_exp = {.i = (integer + 127) << 23};
  float decimal_exp =
    1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
  *dst = int_exp.f * decimal_exp;
}
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_EXP_H_
