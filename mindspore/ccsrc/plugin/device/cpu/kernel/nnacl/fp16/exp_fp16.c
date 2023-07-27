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
#include "nnacl/fp16/exp_fp16.h"
#include <math.h>
#include <string.h>
#include "nnacl/errorcode.h"

#if defined(ENABLE_NEON)
static inline void simd_exp_fp16(float16x8_t input, float16_t *dst) {
  static float16x8_t maxv = {88.72283935546875f, 88.72283935546875f, 88.72283935546875f, 88.72283935546875f,
                             88.72283935546875f, 88.72283935546875f, 88.72283935546875f, 88.72283935546875f};
  static float16x8_t minv = {-87.3365478515625f, -87.3365478515625f, -87.3365478515625f, -87.3365478515625f,
                             -87.3365478515625f, -87.3365478515625f, -87.3365478515625f, -87.3365478515625f};
  input = vmaxq_f16(minv, vminq_f16(input, maxv));
  vst1q_f16(dst, VexpFp16(input));
}
#endif

void ExpFp16(const float16_t *src, float16_t *dst, int num) {
  int i = 0;
#ifdef ENABLE_NEON
  int count = (num / C8NUM) * C8NUM;
  for (; i < count; i += C8NUM) {
    simd_exp_fp16(vld1q_f16(src + i), dst + i);
  }
#endif
  for (; i < num; ++i) {
    single_exp_fp16(src[i], dst + i);
  }
}

int ExpFusionFp16(const void *src_data, void *dst_data, const ExpStruct *exp, int task_id) {
  NNACL_CHECK_ZERO_RETURN_ERR(exp->base_.thread_nr_);
  ExpParameter *param = (ExpParameter *)exp->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  float16_t *src = (float16_t *)src_data;
  float16_t *dst = (float16_t *)dst_data;
  int stride = UP_DIV(exp->element_num_, exp->base_.thread_nr_);
  int start = stride * task_id;
  int end = MSMIN(exp->element_num_, start + stride);
  int num = end - start;

  if (param->scale_ == 1) {
    ExpFp16(src + start, dst + start, num);
  } else {
    int i = 0;
#ifdef ENABLE_ARM64
    MS_FLOAT16X8 scale = MS_MOVQ_F16(exp->in_scale_);
    int count = (num / C8NUM) * C8NUM;
    for (; i < count; i += C8NUM) {
      simd_exp_fp16(MS_MULQ_F16(MS_LDQ_F16(src + i), scale), dst + i);
    }
#endif
    for (; i < num; ++i) {
      single_exp_fp16(src[i] * exp->in_scale_, dst + i);
    }
  }
  if (exp->out_scale_ != 1) {
    int i = 0;
#ifdef ENABLE_ARM64
    MS_FLOAT16X8 scale = MS_MOVQ_F16(exp->out_scale_);
    int count = (num / C8NUM) * C8NUM;
    for (; i < count; i += C8NUM) {
      simd_exp_fp16(MS_LDQ_F16(src + i), dst + i);
      MS_STQ_F16(dst + i, MS_MULQ_F16(MS_LDQ_F16(dst + i), scale));
    }
#endif
    for (; i < num; ++i) {
      single_exp_fp16(src[i], dst + i);
      dst[i] *= exp->out_scale_;
    }
  }
  return NNACL_OK;
}
