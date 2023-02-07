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

#include "nnacl/int8/batch_to_space_int8.h"

void BatchToSpaceNoCropForNHWCInt8(const int8_t *input, int8_t *output, const int32_t *in_shape, int out_n,
                                   const int32_t *block, const QuantArg *in_quant_arg, const QuantArg *out_quant_arg) {
  int block_h = block[0];
  int block_w = block[1];
  int in_h = in_shape[1];
  int in_w = in_shape[2];
  int in_c = in_shape[3];
  int64_t stride_h = block_w * out_n;
  int64_t output_offset = 0;
  int64_t in_stride_h = in_w * in_c;
  int64_t in_stride_n = in_stride_h * in_h;
  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float scale = in_quant_arg->scale_ * output_inverse_scale;
  float bias = -in_quant_arg->zp_ * scale;
  int32_t output_zp = out_quant_arg->zp_;

  for (int n = 0; n < out_n; ++n) {
    for (int h = 0; h < in_h; ++h) {
      int64_t h_offset = h * in_stride_h;
      for (int bh = 0; bh < block_h; ++bh) {
        for (int w = 0; w < in_w; ++w) {
          int64_t w_offset = w * in_c;
          for (int bw = 0; bw < block_w; ++bw) {
            int64_t in_offset = in_stride_n * (bh * stride_h + bw * out_n + n) + w_offset + h_offset;
            for (int c = 0; c < in_c; ++c) {
              int32_t output_tmp = round(input[in_offset + c] * scale + bias) + output_zp;
              output_tmp = output_tmp > 127 ? 127 : output_tmp;
              output_tmp = output_tmp < -128 ? -128 : output_tmp;
              output[output_offset++] = output_tmp;
            }
          }
        }
      }
    }
  }
}

void BatchToSpaceForNHWCInt8(const int8_t *input, int8_t *output, const int32_t *in_shape, int out_n,
                             const int32_t *block, const int32_t *crops, const QuantArg *in_quant_arg,
                             const QuantArg *out_quant_arg) {
  int block_h = block[0];
  int block_w = block[1];
  int in_h = in_shape[1];
  int in_w = in_shape[2];
  int in_c = in_shape[3];
  int h_start = crops[0] / block_h;
  int h_valid_begin = crops[0];
  int h_end = MSMIN((in_h * block_h - crops[1]) / block_h + 1, in_h);
  int h_valid_end = in_h * block_h - crops[1] - 1;
  int w_start = crops[2] / block_w;
  int w_valid_begin = crops[2];
  int w_end = MSMIN((in_w * block_w - crops[3]) / block_w + 1, in_w);
  int w_valid_end = in_w * block_w - crops[3] - 1;

  int64_t stride_h = block_w * out_n;
  int64_t output_offset = 0;
  int64_t in_stride_h = in_w * in_c;
  int64_t in_stride_n = in_stride_h * in_h;

  const float output_inverse_scale = 1.f / out_quant_arg->scale_;
  float scale = in_quant_arg->scale_ * output_inverse_scale;
  float bias = -in_quant_arg->zp_ * scale;
  int32_t output_zp = out_quant_arg->zp_;

  for (int n = 0; n < out_n; ++n) {
    for (int h = h_start; h < h_end; ++h) {
      int64_t h_offset = h * in_stride_h;
      for (int bh = 0; bh < block_h; ++bh) {
        int64_t h_index = h * block_h + bh;
        if (h_index < h_valid_begin || h_index > h_valid_end) {
          continue;
        }
        for (int w = w_start; w < w_end; ++w) {
          int64_t w_offset = w * in_c;
          for (int bw = 0; bw < block_w; ++bw) {
            int64_t w_index = w * block_w + bw;
            if (w_index < w_valid_begin || w_index > w_valid_end) {
              continue;
            }
            int64_t in_offset = in_stride_n * (bh * stride_h + bw * out_n + n) + w_offset + h_offset;
            for (int c = 0; c < in_c; ++c) {
              int32_t output_tmp = round(input[in_offset + c] * scale + bias) + output_zp;
              output_tmp = output_tmp > 127 ? 127 : output_tmp;
              output_tmp = output_tmp < -128 ? -128 : output_tmp;
              output[output_offset++] = output_tmp;
            }
          }
        }
      }
    }
  }
}
