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
#include "nnacl/fp16/layer_norm_fp16.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"

int LayerNormMeanAndSquareFp16(const float16_t *src, int num, float16_t *mean, float16_t *square_mean) {
  if (num <= 0) {
    return NNACL_ERR;
  }
  int index = 0;
  float sum = 0.0f;
  float square_sum = 0.0f;
  for (; index <= num - C8NUM; index += C8NUM) {
    float16x8_t srcv = vld1q_f16(src + index);
    for (int i = 0; i < C8NUM; ++i) {
      square_sum += srcv[i] * srcv[i];
    }
    float16x4_t sum2 = vadd_f16(vget_low_f16(srcv), vget_high_f16(srcv));
    float32x4_t sum_f32 = vcvt_f32_f16(sum2);
    sum += MS_ADDVQ_F32(sum_f32);
  }
  for (; index < num; index++) {
    sum += src[index];
    square_sum += src[index] * src[index];
  }
  *mean = (float16_t)(sum / num);
  *square_mean = (float16_t)(square_sum / num);
  return NNACL_OK;
}

void LayerNormGammaAndBetaFp16(float16_t *dst, const float16_t *src, const float16_t *gamma_data,
                               const float16_t *beta_data, int num, const float16_t mean, const float16_t deno) {
  int index = 0;
  float16x8_t meanv = vdupq_n_f16(mean);
  float16x8_t denov = vdupq_n_f16(deno);
  for (; index <= num - C8NUM; index += C8NUM) {
    float16x8_t srcv = vld1q_f16(src + index);
    float16x8_t outv = vsubq_f16(srcv, meanv);
    outv = vmulq_f16(outv, denov);
    float16x8_t gammav = vld1q_f16(gamma_data + index);
    float16x8_t betav = vld1q_f16(beta_data + index);
    outv = vmulq_f16(outv, gammav);
    outv = vaddq_f16(outv, betav);
    vst1q_f16(dst + index, outv);
  }
  for (; index < num; index++) {
    dst[index] = (src[index] - mean) * (deno);
    dst[index] = dst[index] * gamma_data[index] + beta_data[index];
  }
}

int LayerNormFp16(const float16_t *src_data, const float16_t *gamma_data, const float16_t *beta_data,
                  float16_t *dst_data, float16_t *out_mean, float16_t *out_deno, LayerNormParameter *param,
                  size_t task_id) {
  if (src_data == NULL || dst_data == NULL || gamma_data == NULL || beta_data == NULL) {
    return NNACL_NULL_PTR;
  }
  NNACL_CHECK_ZERO_RETURN_ERR(param->params_inner_size_);
  NNACL_CHECK_ZERO_RETURN_ERR(param->params_outer_size_);
  int step = UP_DIV(param->norm_outer_size_, param->op_parameter_.thread_num_);
  int thread_end = MSMIN((task_id + 1) * step, param->norm_outer_size_);
  for (int i = task_id * step; i < thread_end; i++) {
    const float16_t *src_norm = src_data + i * param->norm_inner_size_;
    float16_t *dst_norm = dst_data + i * param->norm_inner_size_;
    float16_t cur_mean = 0.0f;
    float16_t cur_deno = 0.0f;
    int ret = LayerNormMeanAndSquareFp16(src_norm, param->norm_inner_size_, &cur_mean, &cur_deno);
    if (ret != NNACL_OK) {
      return NNACL_ERR;
    }
    if (out_mean != NULL) {
      out_mean[i] = cur_mean;
    }
    if (out_deno != NULL) {
      out_deno[i] = cur_deno;
    }
    const float16_t deno = 1 / sqrtf(cur_deno - cur_mean * cur_mean + param->epsilon_);
    if (param->norm_outer_size_ <= param->params_outer_size_) {
      for (int x = 0; x < param->norm_inner_size_ / param->params_inner_size_; x++) {
        const float16_t *src_param = src_norm + x * param->params_inner_size_;
        float16_t *dst_param = dst_norm + x * param->params_inner_size_;
        LayerNormGammaAndBetaFp16(dst_param, src_param, gamma_data, beta_data, param->params_inner_size_, cur_mean,
                                  deno);
      }
    } else {
      int x = i / param->params_outer_size_;
      const float16_t *gamma = gamma_data + x * param->norm_inner_size_;
      const float16_t *beta = beta_data + x * param->norm_inner_size_;
      LayerNormGammaAndBetaFp16(dst_norm, src_norm, gamma, beta, param->norm_inner_size_, cur_mean, deno);
    }
  }
  return NNACL_OK;
}
