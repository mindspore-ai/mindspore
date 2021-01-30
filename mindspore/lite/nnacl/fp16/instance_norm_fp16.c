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
#include "nnacl/fp16/instance_norm_fp16.h"
#include <math.h>
#include "nnacl/errorcode.h"

int InstanceNormFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *gamma_data,
                     const float16_t *beta_data, const InstanceNormParameter *param, size_t task_id) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  int channel_step = UP_DIV(param->channel_, param->op_parameter_.thread_num_);
  int channel_begin = task_id * channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, param->channel_);

  for (int b = 0; b < param->batch_; b++) {
    const float16_t *src_b = src_data + b * param->channel_ * param->inner_size_;
    float16_t *dst_b = dst_data + b * param->channel_ * param->inner_size_;
    for (int c = channel_begin; c < channel_end; c++) {
      const float16_t *src = src_b + c * param->inner_size_;
      float16_t *dst = dst_b + c * param->inner_size_;
      float mean = 0.0f;
      float square_mean = 0.0f;

      int index = 0;
      for (; index < param->inner_size_ - C8NUM; index += C8NUM) {
        float16x8_t srcv = vld1q_f16(src + index);
        float16x8_t squarev = vmulq_f16(srcv, srcv);

        float16x4_t sum2 = vadd_f16(vget_low_f16(srcv), vget_high_f16(srcv));
        float32x4_t sum_f32 = vcvt_f32_f16(sum2);
        mean += vaddvq_f32(sum_f32);

        float16x4_t square2 = vadd_f16(vget_low_f16(squarev), vget_high_f16(squarev));
        float32x4_t square_f32 = vcvt_f32_f16(square2);
        square_mean += vaddvq_f32(square_f32);
      }
      for (; index < param->inner_size_; index++) {
        mean += src[index];
        square_mean += src[index] * src[index];
      }

      mean /= (float)param->inner_size_;
      square_mean /= (float)param->inner_size_;
      const float deno = 1 / sqrtf(square_mean - mean * mean + param->epsilon_);

      index = 0;
      float16x8_t meanv = vdupq_n_f16(mean);
      float16x8_t denov = vdupq_n_f16(deno);
      for (; index < param->inner_size_ - C8NUM; index += C8NUM) {
        float16x8_t srcv = vld1q_f16(src + index);
        float16x8_t outv = vsubq_f16(srcv, meanv);
        outv = vmulq_f16(outv, denov);

        float16x8_t gammav = vdupq_n_f16(gamma_data[c]);
        float16x8_t betav = vdupq_n_f16(beta_data[c]);
        outv = vmulq_f16(outv, gammav);
        outv = vaddq_f16(outv, betav);
        vst1q_f16(dst + index, outv);
      }
      for (; index < param->inner_size_; index++) {
        dst[index] = (src[index] - mean) * deno;
        dst[index] = dst[index] * gamma_data[c] + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}
