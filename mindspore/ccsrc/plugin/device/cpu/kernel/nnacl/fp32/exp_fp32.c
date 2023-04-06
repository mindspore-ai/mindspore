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
#include "nnacl/exp_fp32_simd.h"
#include <math.h>
#include <string.h>
#include "nnacl/errorcode.h"

void ExpFp32(const float *src, float *dst, int num) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(ExpFp32, i, src, dst, num);
  for (; i < num; ++i) {
    simd_exp32(src[i], dst + i);
  }
}

int ExpFusionFp32(const void *src_data, void *dst_data, const ExpStruct *exp, int task_id) {
  NNACL_CHECK_ZERO_RETURN_ERR(exp->base_.thread_nr_);
  ExpParameter *param = (ExpParameter *)exp->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  const float *src = (const float *)src_data;
  float *dst = (float *)dst_data;

  int stride = UP_DIV(exp->element_num_, exp->base_.thread_nr_);
  int start = stride * task_id;
  int end = MSMIN(exp->element_num_, start + stride);
  int num = end - start;

  if (param->scale_ == 1) {
    ExpFp32(src + start, dst + start, num);
  } else {
    int i = 0;
    SIMD_RUN_NO_SCALAR(ExpFp32WithInScale, i, src, dst, num, exp->in_scale_);
    for (; i < num; ++i) {
      simd_exp32(src[i] * exp->in_scale_, dst + i);
    }
  }
  if (exp->out_scale_ != 1) {
    int i = 0;
    SIMD_RUN_NO_SCALAR(ExpFp32WithOutScale, i, src, dst, num, exp->out_scale_);
    for (; i < num; ++i) {
      simd_exp32(src[i], dst + i);
      dst[i] *= exp->out_scale_;
    }
  }
  return NNACL_OK;
}
