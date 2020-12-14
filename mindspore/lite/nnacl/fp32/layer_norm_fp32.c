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
#include "nnacl/fp32/layer_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"

int LayerNorm(size_t outer_size, size_t inner_size, const float *src_data, const float *gamma_data,
              const float *beta_data, enum ElementwiseMode elementwise_mode, float epsilon, float *dst_data,
              size_t task_id, size_t thread_num) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }
  if (elementwise_mode != 0 && (gamma_data == NULL || beta_data == NULL)) {
    return NNACL_NULL_PTR;
  }

  for (size_t j = task_id; j < outer_size; j += thread_num) {
    const float *src = src_data + j * inner_size;
    float *dst = dst_data + j * inner_size;
    float mean = 0.0f;
    float square_mean = 0.0f;

    int index = 0;
#ifdef ENABLE_NEON
    float32x4_t sum = vdupq_n_f32(0);
    float32x4_t square_sum = vdupq_n_f32(0);
    for (; index < inner_size - C8NUM; index += C8NUM) {
      float32x4_t srcv1 = vld1q_f32(src + index);
      float32x4_t srcv2 = vld1q_f32(src + index + 4);
      float32x4_t squarev1 = vmulq_f32(srcv1, srcv1);
      float32x4_t squarev2 = vmulq_f32(srcv2, srcv2);
      sum = vaddq_f32(sum, srcv1);
      sum = vaddq_f32(sum, srcv2);
      square_sum = vaddq_f32(square_sum, squarev1);
      square_sum = vaddq_f32(square_sum, squarev2);
    }
    mean = sum[0] + sum[1] + sum[2] + sum[3];
    square_mean = square_sum[0] + square_sum[1] + square_sum[2] + square_sum[3];
#endif
    for (; index < inner_size; index++) {
      mean += src[index];
      square_mean += src[index] * src[index];
    }

    mean /= (float)inner_size;
    square_mean /= (float)inner_size;
    const float deno = 1 / sqrtf(square_mean - mean * mean + epsilon);

    index = 0;
#ifdef ENABLE_NEON
    float32x4_t meanv = vdupq_n_f32(mean);
    float32x4_t denov = vdupq_n_f32(deno);
    if (elementwise_mode != 0) {
      for (; index < inner_size - C8NUM; index += C8NUM) {
        float32x4_t srcv1 = vld1q_f32(src + index);
        float32x4_t srcv2 = vld1q_f32(src + index + 4);
        float32x4_t outv1 = vsubq_f32(srcv1, meanv);
        float32x4_t outv2 = vsubq_f32(srcv2, meanv);
        outv1 = vmulq_f32(outv1, denov);
        outv2 = vmulq_f32(outv2, denov);
        if (elementwise_mode == 1) {
          float32x4_t gammav1 = vdupq_n_f32(gamma_data[j]);
          float32x4_t betav1 = vdupq_n_f32(beta_data[j]);
          outv1 = vmulq_f32(outv1, gammav1);
          outv2 = vmulq_f32(outv2, gammav1);
          outv1 = vaddq_f32(outv1, betav1);
          outv2 = vaddq_f32(outv2, betav1);
        } else {
          float32x4_t gammav1 = vld1q_f32(gamma_data + index);
          float32x4_t gammav2 = vld1q_f32(gamma_data + index + 4);
          float32x4_t betav1 = vld1q_f32(beta_data + index);
          float32x4_t betav2 = vld1q_f32(beta_data + index + 4);
          outv1 = vmulq_f32(outv1, gammav1);
          outv2 = vmulq_f32(outv2, gammav2);
          outv1 = vaddq_f32(outv1, betav1);
          outv2 = vaddq_f32(outv2, betav2);
        }
        vst1q_f32(dst + index, outv1);
        vst1q_f32(dst + index + 4, outv2);
      }
    } else {
      for (; index < inner_size - C8NUM; index += C8NUM) {
        float32x4_t srcv1 = vld1q_f32(src + index);
        float32x4_t srcv2 = vld1q_f32(src + index + 4);
        float32x4_t outv1 = vsubq_f32(srcv1, meanv);
        float32x4_t outv2 = vsubq_f32(srcv2, meanv);
        outv1 = vmulq_f32(outv1, denov);
        outv2 = vmulq_f32(outv2, denov);
        vst1q_f32(dst + index, outv1);
        vst1q_f32(dst + index + 4, outv2);
      }
    }
#endif
    for (; index < inner_size; index++) {
      dst[index] = (src[index] - mean) * deno;
      if (elementwise_mode == 1) {
        dst[index] = dst[index] * gamma_data[j] + beta_data[j];
      } else if (elementwise_mode == 2) {
        dst[index] = dst[index] * gamma_data[index] + beta_data[index];
      }
    }
  }
  return NNACL_OK;
}
