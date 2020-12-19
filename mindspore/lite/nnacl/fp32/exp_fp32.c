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

#include "nnacl/fp32/exp_fp32.h"
#include <math.h>
#include <string.h>
#include "nnacl/errorcode.h"

int Exp(const float *input_data, float *output_data, const ExpParameter *parameter, int task_id) {
  if (parameter->scale_ == 1) {
    for (size_t i = task_id; i < parameter->element_num_; i += parameter->thread_num_) {
      output_data[i] = expf(input_data[i]);
    }
  } else {
    for (size_t i = task_id; i < parameter->element_num_; i += parameter->thread_num_) {
      output_data[i] = expf(input_data[i] * parameter->in_scale_);
    }
  }
  if (parameter->out_scale_ != 1) {
    for (size_t i = task_id; i < parameter->element_num_; i += parameter->thread_num_) {
      output_data[i] = output_data[i] * parameter->out_scale_;
    }
  }
  return NNACL_OK;
}

void ExpFp32(const float *src, float *dst, int num) {
  int i = 0;
  const float param[] = {log(2.0f), 1.0f / 120, 1.0f / 24, 1.0f / 6, 1.0f / 2, 1.0f};
#ifdef ENABLE_ARM64
  float32x4_t maxv = vdupq_n_f32(88.0f);
  float32x4_t minv = vdupq_n_f32(-88.0f);
  float32x4_t param0 = vdupq_n_f32(log(2.0f));
  float32x4_t param1 = vdupq_n_f32(1.0f / 120);
  float32x4_t param2 = vdupq_n_f32(1.0f / 24);
  float32x4_t param3 = vdupq_n_f32(1.0f / 6);
  float32x4_t param4 = vdupq_n_f32(0.5f);
  float32x4_t param5 = vdupq_n_f32(1.0f);
  int count = (num / C4NUM) * C4NUM;
  for (; i < count; i += C4NUM) {
    float32x4_t input4 = vmaxq_f32(minv, vminq_f32(maxv, vld1q_f32(src + i)));
    int32x4_t integer4 = vcvtq_s32_f32(vdivq_f32(input4, param0));
    float32x4_t decimal4 = vsubq_f32(input4, vmulq_f32(vcvtq_f32_s32(integer4), param0));
    int32x4_t int_exp4 = vshlq_s32(vaddq_s32(integer4, vdupq_n_s32(127)), vdupq_n_s32(23));
    vst1q_f32(dst + i, vld1q_f32((float32_t *)(&int_exp4)));
    float32x4_t decimal_exp4 = vaddq_f32(param2, vmulq_f32(decimal4, param1));
    decimal_exp4 = vmulq_f32(decimal4, vaddq_f32(param3, vmulq_f32(decimal4, decimal_exp4)));
    decimal_exp4 = vaddq_f32(param5, vmulq_f32(decimal4, vaddq_f32(param4, decimal_exp4)));
    decimal_exp4 = vaddq_f32(param5, vmulq_f32(decimal4, decimal_exp4));
    vst1q_f32(dst + i, vmulq_f32(vld1q_f32(dst + i), decimal_exp4));
  }
#endif
  for (; i < num; ++i) {
    float input = MSMAX(-88.0f, MSMIN(88.0f, src[i]));
    int integer = input / param[0];
    float decimal = input - integer * param[0];
    int int_exp = (integer + 127) << 23;
    memcpy(dst + i, &int_exp, sizeof(float));
    const float decimal_exp =
      1.0f + decimal * (1.0f + decimal * (0.5f + decimal * (param[3] + decimal * (param[2] + decimal * param[1]))));
    dst[i] *= decimal_exp;
  }
}
