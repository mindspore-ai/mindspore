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
#include "nnacl/int8/depth_to_space_int8.h"
#include <string.h>

void DepthToSpaceForNHWCInt8(const int8_t *input, int8_t *output, const int32_t *in_shape, DepthToSpaceParameter *param,
                             QuantArg *in_quant_arg, QuantArg *out_quant_arg) {
  int32_t block_size = param->block_size_;
  int32_t in_shape_dim2 = in_shape[2];
  int32_t in_shape_dim1 = in_shape[1];
  int64_t copy_size = block_size * param->out_stride_dim2_;
  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float scale = in_quant_arg->scale_ * output_inverse_scale;
  float bias = -in_quant_arg->zp_ * scale;
  int32_t output_zp = out_quant_arg->zp_;
  for (int i = 0; i < in_shape[0]; ++i) {
    int64_t in_offset_n = i * param->in_stride_dim0_;
    int64_t out_offset_n = i * param->out_stride_dim0_;
    for (int j = 0; j < in_shape_dim1; ++j) {
      int64_t in_offset_h = in_offset_n + j * param->in_stride_dim1_;
      int64_t out_offset_h = out_offset_n + j * block_size * param->out_stride_dim1_;
      for (int k = 0; k < in_shape_dim2; ++k) {
        int64_t in_offset_w = in_offset_h + k * param->in_stride_dim2_;
        int64_t out_offset_w = out_offset_h + k * block_size * param->out_stride_dim2_;
        for (int l = 0; l < block_size; ++l) {
          int64_t out_offset = out_offset_w + l * param->out_stride_dim1_;
          int64_t in_offset = in_offset_w + l * block_size * param->out_stride_dim2_;
          for (int m = 0; m < copy_size; ++m) {
            int32_t output_tmp = round(input[in_offset + m] * scale + bias) + output_zp;
            output_tmp = output_tmp > 127 ? 127 : output_tmp;
            output_tmp = output_tmp < -128 ? -128 : output_tmp;
            output[out_offset + m] = output_tmp;
          }
        }
      }
    }
  }
}
