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

#include "wrapper/fp32/dequant_int8_to_fp32_wrapper.h"
#include <stdint.h>
#include <string.h>
void DequantDataPerChannel(const int8_t *quant_src, const DeQuantArg **de_quant_args, size_t de_quant_nums,
                           size_t per_batch_size, float *de_quant_dst) {
  if (per_batch_size == 0) {
    return;
  }
  size_t matrix_size = de_quant_nums / per_batch_size;
  for (int i = 0; i < per_batch_size; i++) {
    const DeQuantArg *de_quant_arg = de_quant_args[i];
    float scale = de_quant_arg->scale;
    int32_t zero_point = de_quant_arg->zeroPoint;
    for (int j = 0; j < matrix_size; j++) {
      de_quant_dst[i * matrix_size + j] = (quant_src[i * matrix_size + j] - zero_point) * scale;
    }
  }
}

void DequantData(const int8_t *quant_src, const DeQuantArg **de_quant_args, size_t de_quant_nums, size_t channels,
                 float *de_quant_dst) {
  if (channels == 0) {
    return;
  }
  size_t per_channel_size = de_quant_nums / channels;
  for (size_t i = 0; i < channels; i++) {
    const DeQuantArg *de_quant_arg = de_quant_args[i];
    float scale = de_quant_arg->scale;
    int32_t zero_point = de_quant_arg->zeroPoint;
    float var_corr = de_quant_arg->var_corr;
    float mean_corr = de_quant_arg->mean_corr;
    if (var_corr < 0 || var_corr > 10) {
      var_corr = 1;
    }
    for (size_t j = 0; j < per_channel_size; j++) {
      float dequant_data = (quant_src[per_channel_size * i + j] - zero_point) * scale;
      de_quant_dst[per_channel_size * i + j] = dequant_data * var_corr + mean_corr;
    }
  }
}

void DequantDataPerTensor(const int8_t *quant_src, const DeQuantArg **de_quant_args, size_t de_quant_nums,
                          float *de_quant_dst) {
  const DeQuantArg *de_quant_arg = de_quant_args[0];
  float *quant_clusters = de_quant_arg->clusters;
  float scale = de_quant_arg->scale;
  int32_t zero_point = de_quant_arg->zeroPoint;
  for (int j = 0; j < de_quant_nums; j++) {
    int8_t quant_data = quant_src[j];
    if (quant_clusters != NULL) {
      if (quant_data > INT8_MAX || quant_data < INT8_MIN) {
        return;
      }
      de_quant_dst[j] = quant_clusters[quant_data - INT8_MIN];
    } else {
      de_quant_dst[j] = (quant_data - zero_point) * scale;
    }
  }
}
