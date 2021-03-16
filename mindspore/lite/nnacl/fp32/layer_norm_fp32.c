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

void LayerNormMeanAndSquare(const float *src, int num, float *mean, float *square_mean) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t sum = vdupq_n_f32(0);
  float32x4_t square_sum = vdupq_n_f32(0);
  for (; index < num - C4NUM; index += C4NUM) {
    float32x4_t srcv = vld1q_f32(src + index);
    float32x4_t squarev = vmulq_f32(srcv, srcv);
    sum = vaddq_f32(sum, srcv);
    square_sum = vaddq_f32(square_sum, squarev);
  }
  *mean = sum[0] + sum[1] + sum[2] + sum[3];
  *square_mean = square_sum[0] + square_sum[1] + square_sum[2] + square_sum[3];
#endif
  for (; index < num; index++) {
    *mean += src[index];
    *square_mean += src[index] * src[index];
  }

  *mean /= (float)num;
  *square_mean /= (float)num;
}

void LayerNormGammaAndBeta(float *dst, const float *src, const float *gamma_data, const float *beta_data, int num,
                           const float mean, const float deno) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t meanv = vdupq_n_f32(mean);
  float32x4_t denov = vdupq_n_f32(deno);
  for (; index < num - C4NUM; index += C4NUM) {
    float32x4_t srcv = vld1q_f32(src + index);
    float32x4_t outv = vsubq_f32(srcv, meanv);
    outv = vmulq_f32(outv, denov);
    float32x4_t gammav = vld1q_f32(gamma_data + index);
    float32x4_t betav = vld1q_f32(beta_data + index);
    outv = vmulq_f32(outv, gammav);
    outv = vaddq_f32(outv, betav);
    vst1q_f32(dst + index, outv);
  }
#endif
  for (; index < num; index++) {
    dst[index] = (src[index] - mean) * (deno);
    dst[index] = dst[index] * gamma_data[index] + beta_data[index];
  }
}

int LayerNorm(const float *src_data, const float *gamma_data, const float *beta_data, float *dst_data, float *out_mean,
              float *out_deno, LayerNormParameter *param, size_t task_id) {
  if (src_data == NULL || dst_data == NULL || gamma_data == NULL || beta_data == NULL || out_mean == NULL ||
      out_deno == NULL) {
    return NNACL_NULL_PTR;
  }
  int step = UP_DIV(param->norm_outer_size_, param->op_parameter_.thread_num_);
  int thread_end = MSMIN((task_id + 1) * step, param->norm_outer_size_);
  for (int i = task_id * step; i < thread_end; i++) {
    const float *src_norm = src_data + i * param->norm_inner_size_;
    float *dst_norm = dst_data + i * param->norm_inner_size_;
    out_mean[i] = 0.0f;
    out_deno[i] = 0.0f;
    LayerNormMeanAndSquare(src_norm, param->norm_inner_size_, &out_mean[i], &out_deno[i]);
    const float deno = 1 / sqrtf(out_deno[i] - out_mean[i] * out_mean[i] + param->epsilon_);
    if (param->norm_outer_size_ <= param->params_outer_size_) {
      for (int x = 0; x < param->norm_inner_size_ / param->params_inner_size_; x++) {
        const float *src_param = src_norm + x * param->params_inner_size_;
        float *dst_param = dst_norm + x * param->params_inner_size_;
        LayerNormGammaAndBeta(dst_param, src_param, gamma_data, beta_data, param->params_inner_size_, out_mean[i],
                              deno);
      }
    } else {
      int x = i / param->params_outer_size_;
      const float *gamma = gamma_data + x * param->norm_inner_size_;
      const float *beta = beta_data + x * param->norm_inner_size_;
      LayerNormGammaAndBeta(dst_norm, src_norm, gamma, beta, param->norm_inner_size_, out_mean[i], deno);
    }
  }
  return NNACL_OK;
}
