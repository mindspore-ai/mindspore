/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "nnacl/int8/split_int8.h"
#include "nnacl/split_parameter.h"
#include <string.h>
#include "nnacl/errorcode.h"

int Int8DoSplit(const int8_t *in_data, int8_t **out_data, const int *input_shape, int offset, int num_unit,
                const SplitParameter *param) {
  if (in_data == NULL || out_data == NULL) {
    return NNACL_ERR;
  }
  const int num_split = param->num_split_;
  const int *split_sizes = param->split_sizes_;
  const int *strides = param->strides_;
  const int split_dim = param->split_dim_;
  int in_stride = strides[split_dim];

  int stride_per_split = in_stride * input_shape[split_dim];
  int split_which = offset % num_split;
  int split_times = offset / num_split;
  const int8_t *src = in_data + split_times * stride_per_split;
  for (int i = 0; i < split_which; i++) {
    src += split_sizes[i] * in_stride;
  }

  const QuantArg in_quant_arg = param->quant_arg_.in_args_;
  float in_scale = in_quant_arg.scale_;
  int32_t in_zp = in_quant_arg.zp_;
  const QuantArg *out_quant_arg = param->quant_arg_.out_args_;

  for (int i = offset; i < offset + num_unit; i++) {
    split_which = i % num_split;
    split_times = i / num_split;
    int copy_size = split_sizes[split_which] * in_stride;
    int8_t *dst = out_data[split_which] + split_times * copy_size;
    float out_scale = out_quant_arg[split_which].scale_;
    int32_t out_zp = out_quant_arg[split_which].zp_;
    if (in_scale == out_scale && in_zp == out_zp) {
      (void)memcpy(dst, src, copy_size * sizeof(int8_t));
    } else {
      float scale = in_scale / out_scale;
      float bias = -in_zp * scale;
      for (int j = 0; j < copy_size; j++) {
        int32_t output_tmp = round(src[j] * scale + bias) + out_zp;
        if (output_tmp > param->quant_arg_.output_activation_max_) {
          dst[j] = param->quant_arg_.output_activation_max_;
        } else if (output_tmp < param->quant_arg_.output_activation_min_) {
          dst[j] = param->quant_arg_.output_activation_min_;
        } else {
          dst[j] = (int8_t)output_tmp;
        }
      }
    }
    src += copy_size;
  }

  return NNACL_OK;
}
