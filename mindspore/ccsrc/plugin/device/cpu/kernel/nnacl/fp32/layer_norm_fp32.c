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

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdLayerNormMeanAndSquareCoreCalc(block_size, block_num, src, num, mean, square_mean, index) \
  do {                                                                                                \
    MS_FLOAT_32xN(block_num) sum_##block_num = MS_MOVN_F32(block_size, 0.0f);                         \
    MS_FLOAT_32xN(block_num) square_sum_##block_num = MS_MOVN_F32(block_size, 0.0f);                  \
    for (int block_max_size = num - block_num + 1; index < block_max_size; index += block_num) {      \
      MS_FLOAT_32xN(block_num) value = MS_LD_F32(block_size, src + index);                            \
      MS_FLOAT_32xN(block_num) square_value = MS_MUL_F32(block_size, value, value);                   \
      sum_##block_num = MS_ADD_F32(block_size, sum_##block_num, value);                               \
      square_sum_##block_num = MS_ADD_F32(block_size, square_sum_##block_num, square_value);          \
    }                                                                                                 \
    *mean += MS_GET_SUM_F32(block_size, sum_##block_num);                                             \
    square_mean += MS_GET_SUM_F32(block_size, square_sum_##block_num);                                \
  } while (0)

int LayerNormMeanAndSquare(const float *src, int num, float *mean, float *variance) {
  if (num <= 0) {
    return NNACL_ERR;
  }
  int index = 0;
  float square_mean = 0.f;

  MS_SIMD_RUN_NO_SCALAR(SimdLayerNormMeanAndSquareCoreCalc, src, num, mean, square_mean, index);

  for (; index < num; index++) {
    *mean += src[index];
    square_mean += src[index] * src[index];
  }
  *mean /= (float)num;
  square_mean /= (float)num;
  *variance = square_mean - (*mean) * (*mean);
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdLayerNormGammaAndBetaCoreCalc(block_size, block_num, dst, src, gamma, beta, num, mean, deno, index) \
  do {                                                                                                          \
    MS_FLOAT_32xN(block_num) mean_##block_num = MS_MOVN_F32(block_size, mean);                                  \
    MS_FLOAT_32xN(block_num) deno_##block_num = MS_MOVN_F32(block_size, deno);                                  \
    for (int block_max_size = num - block_num + 1; index < block_max_size; index += block_num) {                \
      MS_FLOAT_32xN(block_num) value = MS_LD_F32(block_size, src + index);                                      \
      MS_FLOAT_32xN(block_num) out_value = MS_SUB_F32(block_size, value, mean_##block_num);                     \
      out_value = MS_MUL_F32(block_size, out_value, deno_##block_num);                                          \
      out_value = MS_FMADD_F32(block_size, out_value, MS_LD_F32(block_size, gamma + index),                     \
                               MS_LD_F32(block_size, beta + index));                                            \
      MS_ST_F32(block_size, dst + index, out_value);                                                            \
    }                                                                                                           \
  } while (0)

void LayerNormGammaAndBeta(float *dst, const float *src, const float *gamma_data, const float *beta_data, int num,
                           const float mean, const float deno) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdLayerNormGammaAndBetaCoreCalc, dst, src, gamma_data, beta_data, num, mean, deno, index);

  for (; index < num; index++) {
    dst[index] = (src[index] - mean) * (deno);
    dst[index] = dst[index] * gamma_data[index] + beta_data[index];
  }
}

int LayerNorm(const float *src_data, const float *gamma_data, const float *beta_data, float *dst_data, float *out_mean,
              float *out_variance, const LayerNormParameter *param, size_t task_id) {
  if (src_data == NULL || dst_data == NULL || gamma_data == NULL || beta_data == NULL) {
    return NNACL_NULL_PTR;
  }
  NNACL_CHECK_ZERO_RETURN_ERR(param->params_inner_size_);
  NNACL_CHECK_ZERO_RETURN_ERR(param->params_outer_size_);
  int step = UP_DIV(param->norm_outer_size_, param->op_parameter_.thread_num_);
  int thread_end = MSMIN((task_id + 1) * step, param->norm_outer_size_);
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
