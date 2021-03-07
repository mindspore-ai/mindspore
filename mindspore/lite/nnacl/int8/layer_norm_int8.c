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

#include "nnacl/int8/layer_norm_int8.h"

void LayerNormGammaAndBetaInt8(int8_t *dst, const int8_t *src, const float *gamma_data, const float *beta_data,
                               LayerNormQuantArg *quant, int num, const float mean, const float deno) {
  for (int i = 0; i < num; i++) {
    float fp32_src = (src[i] - quant->in_zp_) * quant->in_scale_;
    float fp32_dst = (fp32_src - mean) * deno;
    fp32_dst = fp32_dst * gamma_data[i] + beta_data[i];
    int32_t int32_dst = (int32_t)round(fp32_dst * 1.0 / quant->out_scale_ + quant->out_zp_);
    dst[i] = (int8_t)MSMAX(MSMIN(int32_dst, 127), -128);
  }
}

/*
 * origin : (x-mean) / sqrt(variance + epsilon)  * gamma + beta
 * quant  : (x-mean) / sqrt(sum(x * x) - mean * mean) * gamma + beta
 *
 * */
int LayerNormInt8(const int8_t *src_data, const float *gamma_data, const float *beta_data, int8_t *dst_data,
                  LayerNormParameter *param, LayerNormQuantArg *quant, int task_id) {
  if (src_data == NULL || dst_data == NULL || gamma_data == NULL || beta_data == NULL) {
    return NNACL_NULL_PTR;
  }

  int step = UP_DIV(param->norm_outer_size_, param->op_parameter_.thread_num_);
  int thread_end = MSMIN((task_id + 1) * step, param->norm_outer_size_);
  for (int i = task_id * step; i < thread_end; i++) {
    const int8_t *src_norm = src_data + i * param->norm_inner_size_;
    int8_t *dst_norm = dst_data + i * param->norm_inner_size_;
    float mean = 0.0f;
    float square_mean = 0.0f;
    for (int j = 0; j < param->norm_inner_size_; j++) {
      float float_src = (src_norm[j] - quant->in_zp_) * quant->in_scale_;
      mean += float_src;
      square_mean += float_src * float_src;
    }
    mean /= (float)param->norm_inner_size_;
    square_mean /= (float)param->norm_inner_size_;
    const float deno = 1 / sqrtf(square_mean - mean * mean + param->epsilon_);

    if (param->norm_outer_size_ <= param->params_outer_size_) {
      for (int x = 0; x < param->norm_inner_size_ / param->params_inner_size_; x++) {
        const int8_t *src_param = src_norm + x * param->params_inner_size_;
        int8_t *dst_param = dst_norm + x * param->params_inner_size_;
        LayerNormGammaAndBetaInt8(dst_param, src_param, gamma_data, beta_data, quant, param->norm_inner_size_, mean,
                                  deno);
      }
    } else {
      int x = i / param->params_outer_size_;
      const float *gamma = gamma_data + x * param->norm_inner_size_;
      const float *beta = beta_data + x * param->norm_inner_size_;
      LayerNormGammaAndBetaInt8(dst_norm, src_norm, gamma, beta, quant, param->norm_inner_size_, mean, deno);
    }
  }
  return NNACL_OK;
}
