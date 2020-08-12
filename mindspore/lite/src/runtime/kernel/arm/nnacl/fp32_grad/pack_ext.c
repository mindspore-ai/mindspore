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
#include "nnacl/fp32_grad/pack_ext.h"

static int is_a_ge_zero_and_a_lt_b(int a, int b) { return (unsigned)(a) < (unsigned)(b); }

void im2col_hwc(const float *in_data, float *data_col, ConvParameter *conv_param) {
  const int pad_left = /*conv_param->pad_l_*/ conv_param->pad_w_;
  // const int pad_right =  /*conv_param->pad_r_*/conv_param->pad_w_;
  const int pad_up = /*conv_param->pad_u_*/ conv_param->pad_h_;
  // const int pad_down =   /*conv_param->pad_d/*/conv_param->pad_h_;

  const int stride_h = conv_param->stride_h_;
  const int stride_w = conv_param->stride_w_;

  const int dilation_h = conv_param->dilation_h_;
  const int dilation_w = conv_param->dilation_w_;

  const int kernel_h = conv_param->kernel_h_;
  const int kernel_w = conv_param->kernel_w_;

  const int in_height = conv_param->input_h_;
  const int in_width = conv_param->input_w_;

  const int output_h = conv_param->output_h_;
  const int output_w = conv_param->output_w_;
  const int channels = conv_param->input_channel_ / conv_param->group_;
  const int tot_channels = conv_param->input_channel_;

  int /*channel,*/ kernel_row, kernel_col, output_rows, output_col;

  int row_stride_offset = 0;

  for (output_rows = output_h; output_rows; output_rows--) {
    int col_stride_offset = 0;
    for (output_col = output_w; output_col; output_col--) {
      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        int input_row = -pad_up + kernel_row * dilation_h + row_stride_offset;
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_col = -pad_left + kernel_col * dilation_w + col_stride_offset;

          if (is_a_ge_zero_and_a_lt_b(input_row, in_height) && is_a_ge_zero_and_a_lt_b(input_col, in_width)) {
            const int offset = (input_row * in_width + input_col) * tot_channels;
            memcpy(data_col, in_data + offset, sizeof(float) * channels);
            data_col += channels;
          } else {
            memset(data_col, 0, sizeof(float) * channels);
            data_col += channels;
          }
        }
      }
      col_stride_offset += stride_w;
    }
    row_stride_offset += stride_h;
  }
}

// output matrix is (kernel_h*kernel_w*channels)X(output_h*output_w)
void im2row_hwc(const float *in_data, float *data_row, ConvParameter *conv_param) {
  const int pad_left = /*conv_param->pad_l_*/ conv_param->pad_w_;
  // const int pad_right =  /*conv_param->pad_r_*/conv_param->pad_w_;
  const int pad_up = /*conv_param->pad_u_*/ conv_param->pad_h_;
  // const int pad_down =   /*conv_param->pad_d/*/conv_param->pad_h_;

  const int stride_h = conv_param->stride_h_;
  const int stride_w = conv_param->stride_w_;

  const int dilation_h = conv_param->dilation_h_;
  const int dilation_w = conv_param->dilation_w_;

  const int kernel_h = conv_param->kernel_h_;
  const int kernel_w = conv_param->kernel_w_;

  const int in_height = conv_param->input_h_;
  const int in_width = conv_param->input_w_;

  const int output_h = conv_param->output_h_;
  const int output_w = conv_param->output_w_;
  const int channels = conv_param->input_channel_ / conv_param->group_;
  const int tot_channels = conv_param->input_channel_;

  int channel, kernel_row, kernel_col, output_rows, output_col;

  for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
    for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
      for (channel = 0; channel < channels; channel++) {
        int input_row = -pad_up + kernel_row * dilation_h;
        for (output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, in_height)) {
            for (output_col = output_w; output_col; output_col--) {
              *(data_row++) = 0;
            }
          } else {
            int input_col = -pad_left + kernel_col * dilation_w;
            for (output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, in_width)) {
                const int offset = (input_row * in_width + input_col) * tot_channels + channel;
                *(data_row++) = in_data[offset];
              } else {
                *(data_row++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void col2im_hwc(const float *data_col, float *data_im, ConvParameter *conv_param) {
  const int pad_left = /*conv_param->pad_l_*/ conv_param->pad_w_;
  // const int pad_right =  /*conv_param->pad_r_*/conv_param->pad_w_;
  const int pad_up = /*conv_param->pad_u_*/ conv_param->pad_h_;
  // const int pad_down =   /*conv_param->pad_d/*/conv_param->pad_h_;

  const int stride_h = conv_param->stride_h_;
  const int stride_w = conv_param->stride_w_;

  const int dilation_h = conv_param->dilation_h_;
  const int dilation_w = conv_param->dilation_w_;

  const int kernel_h = conv_param->kernel_h_;
  const int kernel_w = conv_param->kernel_w_;

  const int in_height = conv_param->input_h_;
  const int in_width = conv_param->input_w_;

  const int output_h = conv_param->output_h_;
  const int output_w = conv_param->output_w_;
  const int channels = conv_param->input_channel_ / conv_param->group_;
  const int tot_channels = conv_param->input_channel_;

  int kernel_row, kernel_col, output_rows, output_col;

  int row_stride_offset = 0;

  for (output_rows = output_h; output_rows; output_rows--) {
    int col_stride_offset = 0;
    for (output_col = output_w; output_col; output_col--) {
      for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        int input_row = -pad_up + kernel_row * dilation_h + row_stride_offset;
        for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_col = -pad_left + kernel_col * dilation_w + col_stride_offset;

          if (is_a_ge_zero_and_a_lt_b(input_row, in_height) && is_a_ge_zero_and_a_lt_b(input_col, in_width)) {
            int offset = (input_row * in_width + input_col) * tot_channels;
            float *data_im_ptr = &data_im[offset];
            for (int i = 0; i < channels; i++) {
              data_im_ptr[i] += data_col[i];
            }
          }
          data_col += channels;
        }
      }
      col_stride_offset += stride_w;
    }
    row_stride_offset += stride_h;
  }
}
