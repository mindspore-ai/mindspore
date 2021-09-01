/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

void ExpFp32(const float *src, float *dst, int num) {
  int i = 0;
#ifdef ENABLE_ARM64
  int count = (num / C4NUM) * C4NUM;
  for (; i < count; i += C4NUM) {
    simd_exp(vld1q_f32(src + i), dst + i);
  }
#endif
#ifdef ENABLE_AVX
  int count = (num / C8NUM) * C8NUM;
  for (; i < count; i += C8NUM) {
    simd_exp_avx(_mm256_loadu_ps(src + i), dst + i);
  }
#endif
  for (; i < num; ++i) {
    single_exp(src[i], dst + i);
  }
}

int ExpFusionFp32(const float *src, float *dst, const ExpParameter *param, int task_id) {
  NNACL_CHECK_ZERO_RETURN_ERR(param->op_parameter_.thread_num_);
  int stride = UP_DIV(param->element_num_, param->op_parameter_.thread_num_);
  int start = stride * task_id;
  int end = MSMIN(param->element_num_, start + stride);
  int num = end - start;

  if (param->scale_ == 1) {
    ExpFp32(src + start, dst + start, num);
  } else {
    int i = 0;
#ifdef ENABLE_ARM64
    MS_FLOAT32X4 scale = MS_MOVQ_F32(param->in_scale_);
    int count = (num / C4NUM) * C4NUM;
    for (; i < count; i += C4NUM) {
      simd_exp(MS_MULQ_F32(MS_LDQ_F32(src + i), scale), dst + i);
    }
#endif
    for (; i < num; ++i) {
      single_exp(src[i] * param->in_scale_, dst + i);
    }
  }
  if (param->out_scale_ != 1) {
    int i = 0;
#ifdef ENABLE_ARM64
    MS_FLOAT32X4 scale = {param->out_scale_, param->out_scale_, param->out_scale_, param->out_scale_};
    int count = (num / C4NUM) * C4NUM;
    for (; i < count; i += C4NUM) {
      simd_exp(MS_LDQ_F32(src + i), dst + i);
      MS_STQ_F32(dst + i, MS_MULQ_F32(MS_LDQ_F32(dst + i), scale));
    }
#endif
    for (; i < num; ++i) {
      single_exp(src[i], dst + i);
      dst[i] *= param->out_scale_;
    }
  }
  return NNACL_OK;
}
