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
#include "nnacl/fp32/instance_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int InstanceNorm(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                 const InstanceNormParameter *param, size_t task_id) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int channel_step = UP_DIV(param->channel_, param->op_parameter_.thread_num_);
  int channel_begin = task_id * channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, param->channel_);

  for (int b = 0; b < param->batch_; b++) {
    const float *src_b = src_data + b * param->channel_ * param->inner_size_;
    float *dst_b = dst_data + b * param->channel_ * param->inner_size_;
    for (int c = channel_begin; c < channel_end; c++) {
      const float *src = src_b + c * param->inner_size_;
      float *dst = dst_b + c * param->inner_size_;
      double mean = 0.0f;
      double square_mean = 0.0f;

      int index = 0;
#ifdef ENABLE_NEON
      for (; index < param->inner_size_ - C4NUM; index += C4NUM) {
        float32x4_t srcv = vld1q_f32(src + index);
        float32x4_t squarev = vmulq_f32(srcv, srcv);
#ifdef ENABLE_ARM64
        mean += vaddvq_f32(srcv);
        square_mean += vaddvq_f32(squarev);
#else
        float32x2_t src_add2 = vadd_f32(vget_low_f32(srcv), vget_high_f32(srcv));
        float32x2_t src_add4 = vpadd_f32(src_add2, src_add2);
        mean += vget_lane_f32(src_add4, 0);
        float32x2_t square_add2 = vadd_f32(vget_low_f32(squarev), vget_high_f32(squarev));
        float32x2_t square_add4 = vpadd_f32(square_add2, square_add2);
        square_mean += vget_lane_f32(square_add4, 0);
#endif
      }
#endif
      for (; index < param->inner_size_; index++) {
        mean += src[index];
        square_mean += src[index] * src[index];
      }

      mean /= (float)param->inner_size_;
      square_mean /= (float)param->inner_size_;
      const double deno = gamma_data[c] / sqrt(square_mean - mean * mean + param->epsilon_);

      index = 0;
      for (; index < param->inner_size_; index++) {
        dst[index] = (src[index] - mean) * deno + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}
