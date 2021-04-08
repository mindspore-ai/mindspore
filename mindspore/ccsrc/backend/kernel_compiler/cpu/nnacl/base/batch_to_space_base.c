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

#include "nnacl/base/batch_to_space_base.h"

void BatchToSpaceNoCropForNHWC(const void *input, void *output, const int *in_shape, int out_n, const int *block,
                               int data_size) {
  int block_h = block[0];
  int block_w = block[1];
  int in_h = in_shape[1];
  int in_w = in_shape[2];
  int in_c = in_shape[3];
  size_t stride_h = block_w * out_n;
  size_t output_offset = 0;
  size_t copy_size = in_c * data_size;
  size_t in_stride_h = in_w * in_c;
  size_t in_stride_n = in_stride_h * in_h;
  for (int n = 0; n < out_n; ++n) {
    for (int h = 0; h < in_h; ++h) {
      size_t h_offset = h * in_stride_h;
      for (int bh = 0; bh < block_h; ++bh) {
        for (int w = 0; w < in_w; ++w) {
          size_t w_offset = w * in_c;
          for (int bw = 0; bw < block_w; ++bw) {
            size_t in_offset = in_stride_n * (bh * stride_h + bw * out_n + n) + w_offset + h_offset;
            memcpy((int8_t *)output + output_offset, (int8_t *)input + in_offset * data_size, copy_size);
            output_offset += copy_size;
          }
        }
      }
    }
  }
}

void BatchToSpaceForNHWC(const void *input, void *output, const int *in_shape, int out_n, const int *block,
                         const int *crops, int data_size) {
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

  size_t stride_h = block_w * out_n;
  size_t output_offset = 0;
  size_t copy_size = in_c * data_size;
  size_t in_stride_h = in_w * in_c;
  size_t in_stride_n = in_stride_h * in_h;
  for (int n = 0; n < out_n; ++n) {
    for (int h = h_start; h < h_end; ++h) {
      size_t h_offset = h * in_stride_h;
      for (int bh = 0; bh < block_h; ++bh) {
        size_t h_index = h * block_h + bh;
        if (h_index < h_valid_begin || h_index > h_valid_end) {
          continue;
        }
        for (int w = w_start; w < w_end; ++w) {
          size_t w_offset = w * in_c;
          for (int bw = 0; bw < block_w; ++bw) {
            size_t w_index = w * block_w + bw;
            if (w_index < w_valid_begin || w_index > w_valid_end) {
              continue;
            }
            size_t in_offset = in_stride_n * (bh * stride_h + bw * out_n + n) + w_offset + h_offset;
            memcpy((int8_t *)output + output_offset, (int8_t *)input + in_offset * data_size, copy_size);
            output_offset += copy_size;
          }
        }
      }
    }
  }
}
