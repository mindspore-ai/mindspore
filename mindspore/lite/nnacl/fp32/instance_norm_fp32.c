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
#include "nnacl/fp32/instance_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int InstanceNorm(const int outer_size, const int inner_size, const float *src_data, const float *scale_data,
                 const float *bias_data, InstanceNormParameter *param, float *dst_data, const int task_id,
                 const int thread_num) {
  if (src_data == NULL || dst_data == NULL || scale_data == NULL || bias_data == NULL) {
    return NNACL_NULL_PTR;
  }
  for (int j = task_id; j < outer_size; j += thread_num) {
    int offset = (j / param->channel_) * inner_size * param->channel_;
    const float *src = src_data + offset;
    float *dst = dst_data + offset;
    float mean = 0.0f;
    float square_mean = 0.0f;
    for (int i = 0; i < inner_size; i++) {
      int idx = j % param->channel_ + i * param->channel_;
      mean += src[idx];
      square_mean += src[idx] * src[idx];
    }
    mean /= (float)inner_size;
    square_mean /= (float)inner_size;
    const float deno = 1 / sqrtf(square_mean - mean * mean + param->epsilon_);
    for (int i = 0; i < inner_size; ++i) {
      int idx = j % param->channel_ + i * param->channel_;
      int scale_idx = (j / param->channel_) * param->channel_ + j % param->channel_;
      dst[idx] = ((src[idx] - mean) * deno) * scale_data[scale_idx] + bias_data[scale_idx];
    }
  }
  return NNACL_OK;
}
