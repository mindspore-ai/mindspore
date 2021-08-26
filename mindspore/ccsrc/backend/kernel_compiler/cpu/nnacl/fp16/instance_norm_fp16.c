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
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"

int InstanceNormFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *gamma_data,
                     const float16_t *beta_data, const InstanceNormParameter *param, size_t task_id) {
  NNACL_CHECK_NULL_RETURN_ERR(src_data);
  NNACL_CHECK_NULL_RETURN_ERR(dst_data);
  int channel = param->channel_;
  int hw_plane = param->inner_size_;
  int channel_step = UP_DIV(channel, param->op_parameter_.thread_num_);
  int channel_begin = task_id * channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, channel);

  for (int b = 0; b < param->batch_; b++) {
    const float16_t *src_b = src_data + b * channel * hw_plane;
    float16_t *dst_b = dst_data + b * channel * hw_plane;
    for (int c = channel_begin; c < channel_end; c++) {
      const float16_t *src = src_b + c * hw_plane;
      float16_t *dst = dst_b + c * hw_plane;
      float mean = 0.0f;
      float square_mean = 0.0f;

      int index = 0;
      for (; index <= hw_plane - C8NUM; index += C8NUM) {
        float16x8_t srcv = vld1q_f16(src + index);
        float16x8_t squarev = vmulq_f16(srcv, srcv);

        float16x4_t sum2 = vadd_f16(vget_low_f16(srcv), vget_high_f16(srcv));
        float32x4_t sum_f32 = vcvt_f32_f16(sum2);
        mean += MS_ADDVQ_F32(sum_f32);

        float16x4_t square2 = vadd_f16(vget_low_f16(squarev), vget_high_f16(squarev));
        float32x4_t square_f32 = vcvt_f32_f16(square2);
        square_mean += MS_ADDVQ_F32(square_f32);
      }
      for (; index < hw_plane; index++) {
        mean += src[index];
        square_mean += src[index] * src[index];
      }

      mean /= (float)hw_plane;
      square_mean /= (float)hw_plane;
      const float deno = 1 / sqrtf(square_mean - mean * mean + param->epsilon_);

      index = 0;
      float16x8_t meanv = vdupq_n_f16(mean);
      float16x8_t denov = vdupq_n_f16(deno);
      for (; index <= hw_plane - C8NUM; index += C8NUM) {
        float16x8_t srcv = vld1q_f16(src + index);
        float16x8_t outv = vsubq_f16(srcv, meanv);
        outv = vmulq_f16(outv, denov);

        float16x8_t gammav = vdupq_n_f16(gamma_data[c]);
        float16x8_t betav = vdupq_n_f16(beta_data[c]);
        outv = vmulq_f16(outv, gammav);
        outv = vaddq_f16(outv, betav);
        vst1q_f16(dst + index, outv);
      }
      for (; index < hw_plane; index++) {
        dst[index] = (src[index] - mean) * deno;
        dst[index] = dst[index] * gamma_data[c] + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}

int InstanceNormNC8HW8Fp16(const float16_t *src_data, float16_t *dst_data, const float16_t *gamma_data,
                           const float16_t *beta_data, const InstanceNormParameter *param, size_t task_id) {
  NNACL_CHECK_NULL_RETURN_ERR(src_data);
  NNACL_CHECK_NULL_RETURN_ERR(dst_data);
  int channel = param->channel_;
  int hw_plane = param->inner_size_;
  int channel_step = UP_DIV(UP_DIV(channel, C8NUM), param->op_parameter_.thread_num_) * C8NUM;
  int channel_begin = (int)(task_id)*channel_step;
  int channel_end = MSMIN(channel_begin + channel_step, channel);
  int c8_down = channel_end / C8NUM * C8NUM;
  int c_res = channel_end - c8_down;
  float32x4_t hw_plane_4 = vdupq_n_f32(hw_plane);
  for (int b = 0; b < param->batch_; b++) {
    const float16_t *src_b = src_data + b * channel * hw_plane;
    float16_t *dst_b = dst_data + b * channel * hw_plane;
    int c = channel_begin;
    for (; c <= channel_end - C16NUM; c += C16NUM) {
      const float16_t *src = src_b + c * hw_plane;
      const float16_t *src1 = src_b + (c + C8NUM) * hw_plane;
      float16_t *dst = dst_b + c;
      float32x4_t mean1 = vdupq_n_f32(0.0f);
      float32x4_t mean2 = vdupq_n_f32(0.0f);
      float32x4_t mean3 = vdupq_n_f32(0.0f);
      float32x4_t mean4 = vdupq_n_f32(0.0f);
      float32x4_t square_mean1 = vdupq_n_f32(0.0f);
      float32x4_t square_mean2 = vdupq_n_f32(0.0f);
      float32x4_t square_mean3 = vdupq_n_f32(0.0f);
      float32x4_t square_mean4 = vdupq_n_f32(0.0f);
      for (int index = 0; index < hw_plane; ++index) {
        float16x8_t srcv = vld1q_f16(src + index * C8NUM);
        float16x8_t srcv1 = vld1q_f16(src1 + index * C8NUM);

        float32x4_t srcv01 = vcvt_f32_f16(vget_low_f16(srcv));
        float32x4_t srcv02 = vcvt_f32_f16(vget_high_f16(srcv1));
        float32x4_t srcv11 = vcvt_f32_f16(vget_low_f16(srcv));
        float32x4_t srcv12 = vcvt_f32_f16(vget_high_f16(srcv1));
        mean1 = vaddq_f32(mean1, srcv01);
        mean2 = vaddq_f32(mean2, srcv02);
        mean3 = vaddq_f32(mean3, srcv11);
        mean4 = vaddq_f32(mean4, srcv12);
        square_mean1 = vaddq_f32(square_mean1, vmulq_f32(srcv01, srcv01));
        square_mean2 = vaddq_f32(square_mean2, vmulq_f32(srcv02, srcv02));
        square_mean3 = vaddq_f32(square_mean3, vmulq_f32(srcv11, srcv11));
        square_mean4 = vaddq_f32(square_mean4, vmulq_f32(srcv12, srcv12));
      }
      float16x8_t mean =
        vcombine_f16(vcvt_f16_f32(MS_DIVQ_F32(mean1, hw_plane_4)), vcvt_f16_f32(MS_DIVQ_F32(mean2, hw_plane_4)));
      float16x8_t mean_1 =
        vcombine_f16(vcvt_f16_f32(MS_DIVQ_F32(mean3, hw_plane_4)), vcvt_f16_f32(MS_DIVQ_F32(mean4, hw_plane_4)));
      float16x8_t square_mean = vcombine_f16(vcvt_f16_f32(MS_DIVQ_F32(square_mean1, hw_plane_4)),
                                             vcvt_f16_f32(MS_DIVQ_F32(square_mean2, hw_plane_4)));
      float16x8_t square_mean_1 = vcombine_f16(vcvt_f16_f32(MS_DIVQ_F32(square_mean3, hw_plane_4)),
                                               vcvt_f16_f32(MS_DIVQ_F32(square_mean4, hw_plane_4)));
      float16x8_t deno = vaddq_f16(vsubq_f16(square_mean, vmulq_f16(mean, mean)), vdupq_n_f16(param->epsilon_));
      float16x8_t deno1 = vaddq_f16(vsubq_f16(square_mean_1, vmulq_f16(mean_1, mean_1)), vdupq_n_f16(param->epsilon_));
      deno = 1 / MS_SQRTFX8_F16(deno);
      deno1 = 1 / MS_SQRTFX8_F16(deno1);

      float16x8_t gammav = vmulq_f16(vld1q_f16(gamma_data + c), deno);            // deno * gamma_data[c]
      float16x8_t gammav1 = vmulq_f16(vld1q_f16(gamma_data + c + C8NUM), deno1);  // deno * gamma_data[c]
      float16x8_t betav = vld1q_f16(beta_data + c);
      float16x8_t betav1 = vld1q_f16(beta_data + c + C8NUM);
      for (int index = 0; index < hw_plane; ++index) {
        float16x8_t srcv = vld1q_f16(src + index * C8NUM);
        float16x8_t srcv1 = vld1q_f16(src1 + index * C8NUM);
        float16x8_t outv = vsubq_f16(srcv, mean);
        float16x8_t outv1 = vsubq_f16(srcv1, mean1);
        outv = vmulq_f16(outv, gammav);
        outv1 = vmulq_f16(outv1, gammav1);
        outv = vaddq_f16(outv, betav);
        outv1 = vaddq_f16(outv1, betav1);
        vst1q_f16(dst + index * channel, outv);
        vst1q_f16(dst + index * channel + C8NUM, outv1);
      }
    }
    for (; c <= channel_end - C8NUM; c += C8NUM) {
      const float16_t *src = src_b + c * hw_plane;
      float16_t *dst = dst_b + c;
      float32x4_t mean1 = vdupq_n_f32(0.0f);
      float32x4_t mean2 = vdupq_n_f32(0.0f);
      float32x4_t square_mean1 = vdupq_n_f32(0.0f);
      float32x4_t square_mean2 = vdupq_n_f32(0.0f);
      for (int index = 0; index < hw_plane; ++index) {
        float16x8_t srcv = vld1q_f16(src + index * C8NUM);
        float32x4_t srcv1 = vcvt_f32_f16(vget_low_f16(srcv));
        float32x4_t srcv2 = vcvt_f32_f16(vget_high_f16(srcv));
        mean1 = vaddq_f32(mean1, srcv1);
        mean2 = vaddq_f32(mean2, srcv2);
        square_mean1 = vaddq_f32(square_mean1, vmulq_f32(srcv1, srcv1));
        square_mean2 = vaddq_f32(square_mean2, vmulq_f32(srcv2, srcv2));
      }
      float16x8_t mean =
        vcombine_f16(vcvt_f16_f32(MS_DIVQ_F32(mean1, hw_plane_4)), vcvt_f16_f32(MS_DIVQ_F32(mean2, hw_plane_4)));
      float16x8_t square_mean = vcombine_f16(vcvt_f16_f32(MS_DIVQ_F32(square_mean1, hw_plane_4)),
                                             vcvt_f16_f32(MS_DIVQ_F32(square_mean2, hw_plane_4)));
      float16x8_t deno =
        vaddq_f16(vsubq_f16(square_mean, vmulq_f16(mean, mean)), vdupq_n_f16(param->epsilon_));  // question
      deno = 1 / MS_SQRTFX8_F16(deno);                                                           // question

      float16x8_t gammav = vmulq_f16(vld1q_f16(gamma_data + c), deno);  // deno * gamma_data[c]
      float16x8_t betav = vld1q_f16(beta_data + c);
      for (int index = 0; index < hw_plane; ++index) {
        float16x8_t srcv = vld1q_f16(src + index * C8NUM);
        float16x8_t outv = vsubq_f16(srcv, mean);
        outv = vmulq_f16(outv, gammav);
        outv = vaddq_f16(outv, betav);
        vst1q_f16(dst + index * channel, outv);
      }
    }
    for (; c < channel_end; ++c) {
      const float16_t *src = src_b + c8_down * hw_plane + c;
      float16_t *dst = dst_b + c;
      float mean = 0.0f;
      float square_mean = 0.0f;
      for (int index = 0; index < hw_plane; ++index) {
        float16_t tmp = src[index * c_res];
        mean += tmp;
        square_mean += tmp * tmp;
      }
      mean /= (float)hw_plane;
      square_mean /= (float)hw_plane;
      const float deno = gamma_data[c] / sqrtf(square_mean - mean * mean + param->epsilon_);
      for (int index = 0; index < hw_plane; ++index) {
        dst[index * channel] = (src[index * c_res] - mean) * deno + beta_data[c];
      }
    }
  }
  return NNACL_OK;
}
