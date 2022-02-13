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
  static float16x8_t maxv = {88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f, 88.0f};
  static float16x8_t minv = {-88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f, -88.0f};
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

int ExpFusionFp16(const float16_t *src, float16_t *dst, const ExpParameter *param, int task_id) {
  NNACL_CHECK_ZERO_RETURN_ERR(param->op_parameter_.thread_num_);
  int stride = UP_DIV(param->element_num_, param->op_parameter_.thread_num_);
  int start = stride * task_id;
  int end = MSMIN(param->element_num_, start + stride);
  int num = end - start;

  if (param->scale_ == 1) {
    ExpFp16(src + start, dst + start, num);
  } else {
    int i = 0;
#ifdef ENABLE_ARM64
    MS_FLOAT16X8 scale = MS_MOVQ_F16(param->in_scale_);
    int count = (num / C8NUM) * C8NUM;
    for (; i < count; i += C8NUM) {
      simd_exp_fp16(MS_MULQ_F16(MS_LDQ_F16(src + i), scale), dst + i);
    }
#endif
    for (; i < num; ++i) {
      single_exp_fp16(src[i] * param->in_scale_, dst + i);
    }
  }
  if (param->out_scale_ != 1) {
    int i = 0;
#ifdef ENABLE_ARM64
    MS_FLOAT16X8 scale = MS_MOVQ_F16(param->out_scale_);
    int count = (num / C8NUM) * C8NUM;
    for (; i < count; i += C8NUM) {
      simd_exp_fp16(MS_LDQ_F16(src + i), dst + i);
      MS_STQ_F16(dst + i, MS_MULQ_F16(MS_LDQ_F16(dst + i), scale));
    }
#endif
    for (; i < num; ++i) {
      single_exp_fp16(src[i], dst + i);
      dst[i] *= param->out_scale_;
    }
  }
  return NNACL_OK;
}
