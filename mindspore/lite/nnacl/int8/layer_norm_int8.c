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

/*
 * origin : (x-mean) / sqrt(variance + epsilon)  * gamma + beta
 * quant  : (x-mean) / sqrt(sum(x * x) - mean * mean) * gamma + beta
 *
 * */
int LayerNormInt8(const int8_t *src_data, const float *gamma_data, const float *beta_data, int8_t *dst_data,
                  enum ElementwiseMode elementwise_mode, int outer_size, int inner_size, LayerNormQuantArg *quant,
                  float epsilon) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }

  if (elementwise_mode != 0 && (gamma_data == NULL || beta_data == NULL)) {
    return NNACL_NULL_PTR;
  }

  for (int out_index = 0; out_index < outer_size; out_index++) {
    const int8_t *src = src_data + out_index * inner_size;
    int8_t *dst = dst_data + out_index * inner_size;
    float mean = 0.0f;
    float square_mean = 0.0f;
    for (int i = 0; i < inner_size; i++) {
      float float_src = (src[i] - quant->in_zp_) * quant->in_scale_;
      mean += float_src;
      square_mean += float_src * float_src;
    }
    mean /= (float)inner_size;
    square_mean /= (float)inner_size;
    const float deno = 1 / sqrtf(square_mean - mean * mean + epsilon);
    for (int i = 0; i < inner_size; i++) {
      float fp32_src = (src[i] - quant->in_zp_) * quant->in_scale_;
      float fp32_dst = (fp32_src - mean) * deno;
      if (elementwise_mode == 1) {
        fp32_dst = fp32_dst * gamma_data[out_index] + beta_data[out_index];
      } else if (elementwise_mode == 2) {
        fp32_dst = fp32_dst * gamma_data[i] + beta_data[i];
      }
      int32_t int32_dst = (int32_t)round(fp32_dst * 1.0 / quant->out_scale_ + quant->out_zp_);
      dst[i] = (int8_t)MSMAX(MSMIN(int32_dst, 127), -128);
    }
  }
  return NNACL_OK;
}
