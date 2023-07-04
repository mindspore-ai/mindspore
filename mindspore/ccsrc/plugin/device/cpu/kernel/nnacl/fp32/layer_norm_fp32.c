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
#include "nnacl/fp32/layer_norm_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/layer_norm_fp32_simd.h"

int LayerNormMeanAndSquare(const float *src, int num, float *mean, float *variance) {
  if (num <= 0) {
    return NNACL_ERR;
  }
  int index = 0;
  float square_mean = 0.f;

  SIMD_RUN_NO_SCALAR(LayerNormMeanAndSquare, index, src, num, mean, &square_mean);

  for (; index < num; index++) {
    *mean += src[index];
    square_mean += src[index] * src[index];
  }
  *mean /= (float)num;
  square_mean /= (float)num;
  *variance = square_mean - (*mean) * (*mean);
  return NNACL_OK;
}

void LayerNormGammaAndBeta(float *dst, const float *src, const float *gamma_data, const float *beta_data, int num,
                           const float mean, const float deno) {
  int index = 0;

  SIMD_RUN_NO_SCALAR(LayerNormGammaAndBeta, index, dst, src, gamma_data, beta_data, num, mean, deno);

  for (; index < num; index++) {
    dst[index] = (src[index] - mean) * (deno);
    dst[index] = dst[index] * gamma_data[index] + beta_data[index];
  }
}

int LayerNorm(const float *src_data, const float *gamma_data, const float *beta_data, float *dst_data, float *out_mean,
              float *out_variance, const LayerNormComputeParam *param, int task_id, int thread_num) {
  if (src_data == NULL || dst_data == NULL || gamma_data == NULL || beta_data == NULL) {
    return NNACL_NULL_PTR;
  }
  NNACL_CHECK_NULL_RETURN_ERR(param);
  NNACL_CHECK_ZERO_RETURN_ERR(param->params_inner_size_);
  NNACL_CHECK_ZERO_RETURN_ERR(param->params_outer_size_);
  int step = UP_DIV(param->norm_outer_size_, thread_num);
  int thread_end = MSMIN(((int)task_id + 1) * step, param->norm_outer_size_);
  for (int i = task_id * step; i < thread_end; i++) {
    const float *src_norm = src_data + i * param->norm_inner_size_;
    float *dst_norm = dst_data + i * param->norm_inner_size_;
    float cur_mean = 0.0f;
    float cur_variance = 0.0f;
    int ret = LayerNormMeanAndSquare(src_norm, param->norm_inner_size_, &cur_mean, &cur_variance);
    if (ret != NNACL_OK) {
      return NNACL_ERR;
    }
    if (out_mean != NULL) {
      out_mean[i] = cur_mean;
    }
    if (out_variance != NULL) {
      out_variance[i] = cur_variance;
    }
    const float deno = 1 / sqrtf(cur_variance + param->epsilon_);
    if (param->norm_outer_size_ <= param->params_outer_size_) {
      for (int x = 0; x < param->norm_inner_size_ / param->params_inner_size_; x++) {
        const float *src_param = src_norm + x * param->params_inner_size_;
        float *dst_param = dst_norm + x * param->params_inner_size_;
        LayerNormGammaAndBeta(dst_param, src_param, gamma_data, beta_data, param->params_inner_size_, cur_mean, deno);
      }
    } else {
      int x = i / param->params_outer_size_;
      const float *gamma = gamma_data + x * param->norm_inner_size_;
      const float *beta = beta_data + x * param->norm_inner_size_;
      LayerNormGammaAndBeta(dst_norm, src_norm, gamma, beta, param->norm_inner_size_, cur_mean, deno);
    }
  }
  return NNACL_OK;
}
