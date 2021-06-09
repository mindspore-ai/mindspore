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

#include <string.h>
#include "nnacl/fp16_grad/pack_fp16_ext.h"

void RollingIm2ColPackDwUnitFp16(const float16_t *in_data, const ConvParameter *conv_param, float16_t *data_col_orig,
                                 int real_cal_num, int start) {
  const int pad_left = conv_param->pad_l_;
  const int pad_up = conv_param->pad_u_;

  const int stride_h = conv_param->stride_h_;
  const int stride_w = conv_param->stride_w_;

  const int dilation_h = conv_param->dilation_h_;
  const int dilation_w = conv_param->dilation_w_;

  const int kernel_h = conv_param->kernel_h_;
  const int kernel_w = conv_param->kernel_w_;

  const int in_height = conv_param->input_h_;
  const int in_width = conv_param->input_w_;

  const int output_w = conv_param->output_w_;

  const int channels = conv_param->input_channel_;
  const int stride = kernel_h * kernel_w;

  int kernel_row, kernel_col;

  for (int i = 0; i < real_cal_num; i++) {
    int block_start = start + i;
    int input_h = block_start / output_w * stride_h;
    int input_w = block_start % output_w * stride_w;
    float16_t *data_col = data_col_orig + i * channels * stride;
    for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      int input_row = -pad_up + kernel_row * dilation_h + input_h;
      for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_col = -pad_left + kernel_col * dilation_w + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_height)) && ((unsigned)(input_col) < (unsigned)(in_width))) {
          const int offset = (input_row * in_width + input_col) * channels;
          for (int c = 0; c < channels; c++) {
            data_col[c * stride] = in_data[offset + c];
          }
          data_col++;
        } else {
          for (int c = 0; c < channels; c++) {
            data_col[c * stride] = 0;
          }
          data_col++;
        }
      }
    }
  }
}

void RollingIm2ColPackUnitFp16(const float16_t *input_data, const ConvParameter *conv_param, float16_t *packed_input,
                               int real_cal_num, int block_index) {
  const int pad_left = conv_param->pad_l_;
  const int pad_up = conv_param->pad_u_;

  const int stride_h = conv_param->stride_h_;
  const int stride_w = conv_param->stride_w_;

  const int dilation_h = conv_param->dilation_h_;
  const int dilation_w = conv_param->dilation_w_;

  const int kernel_h = conv_param->kernel_h_;
  const int kernel_w = conv_param->kernel_w_;

  const int in_height = conv_param->input_h_;
  const int in_width = conv_param->input_w_;

  const int output_w = conv_param->output_w_;

  const int channels = conv_param->input_channel_ / conv_param->group_;
  const int tot_channels = conv_param->input_channel_;

  int kernel_row, kernel_col;

  if (channels == 1) {
    for (int i = 0; i < real_cal_num; i++) {
      int block_start = block_index + i;
      int input_h = block_start / output_w * stride_h;
      int input_w = block_start % output_w * stride_w;
      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        int input_row = -pad_up + kernel_row * dilation_h + input_h;
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_col = -pad_left + kernel_col * dilation_w + input_w;
          if (((unsigned)(input_row) < (unsigned)(in_height)) && ((unsigned)(input_col) < (unsigned)(in_width))) {
            const int offset = (input_row * in_width + input_col) * tot_channels;
            *packed_input = input_data[offset];
            packed_input++;
          } else {
            *packed_input = 0;
            packed_input++;
          }
        }
      }
    }
  } else {
    for (int i = 0; i < real_cal_num; i++) {
      int block_start = block_index + i;
      int input_h = block_start / output_w * stride_h;
      int input_w = block_start % output_w * stride_w;
      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        int input_row = -pad_up + kernel_row * dilation_h + input_h;
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_col = -pad_left + kernel_col * dilation_w + input_w;
          if (((unsigned)(input_row) < (unsigned)(in_height)) && ((unsigned)(input_col) < (unsigned)(in_width))) {
            const int offset = (input_row * in_width + input_col) * tot_channels;
            memcpy(packed_input, input_data + offset, sizeof(float16_t) * channels);
            packed_input += channels;
          } else {
            memset(packed_input, 0, sizeof(float16_t) * channels);
            packed_input += channels;
          }
        }
      }
    }
  }
}

void RollingCol2ImPackUnitFp16(const float16_t *data_col, float16_t *data_im, const ConvParameter *conv_param,
                               int real_cal_num, int block_index) {
  const int pad_left = conv_param->pad_l_;
  const int pad_up = conv_param->pad_u_;

  const int stride_h = conv_param->stride_h_;
  const int stride_w = conv_param->stride_w_;

  const int dilation_h = conv_param->dilation_h_;
  const int dilation_w = conv_param->dilation_w_;

  const int kernel_h = conv_param->kernel_h_;
  const int kernel_w = conv_param->kernel_w_;

  const int in_height = conv_param->input_h_;
  const int in_width = conv_param->input_w_;

  const int output_w = conv_param->output_w_;
  const int channels = conv_param->input_channel_ / conv_param->group_;
  const int tot_channels = conv_param->input_channel_;

  int kernel_row, kernel_col;

  if (channels == 1) {
    for (int r = 0; r < real_cal_num; r++) {
      int output_col = (block_index + r) % output_w;
      int output_row = (block_index + r) / output_w;
      int row_stride_offset = output_row * stride_h;
      int col_stride_offset = output_col * stride_w;
      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        int input_row = -pad_up + kernel_row * dilation_h + row_stride_offset;
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_col = -pad_left + kernel_col * dilation_w + col_stride_offset;
          if (((unsigned)(input_row) < (unsigned)(in_height)) && ((unsigned)(input_col) < (unsigned)(in_width))) {
            int offset = (input_row * in_width + input_col) * tot_channels;
            float16_t *data_im_ptr = data_im + offset;
            *data_im_ptr += *data_col;
          }
          data_col++;
        }
      }
    }
  } else {
    for (int r = 0; r < real_cal_num; r++) {
      int output_col = (block_index + r) % output_w;
      int output_row = (block_index + r) / output_w;
      int row_stride_offset = output_row * stride_h;
      int col_stride_offset = output_col * stride_w;
      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        int input_row = -pad_up + kernel_row * dilation_h + row_stride_offset;
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_col = -pad_left + kernel_col * dilation_w + col_stride_offset;
          if (((unsigned)(input_row) < (unsigned)(in_height)) && ((unsigned)(input_col) < (unsigned)(in_width))) {
            int offset = (input_row * in_width + input_col) * tot_channels;
            float16_t *data_im_ptr = &data_im[offset];
            for (int i = 0; i < channels; i++) {
              data_im_ptr[i] += data_col[i];
            }
          }
          data_col += channels;
        }
      }
    }
  }
}
