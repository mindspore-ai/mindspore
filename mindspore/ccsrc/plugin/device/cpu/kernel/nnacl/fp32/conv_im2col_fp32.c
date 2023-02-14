/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/conv_im2col_fp32.h"

void Im2ColDataPackUnitFp32(const float *input_data, const ConvParameter *conv_param, float *packed_input,
                            int real_cal_num, int block_index) {
  // input format : nhwc
  int kernel_w = conv_param->kernel_w_;
  int kernel_h = conv_param->kernel_h_;
  int kernel_plane = kernel_h * kernel_w;
  int dilation_w = conv_param->dilation_w_;
  int dilation_h = conv_param->dilation_h_;

  int out_w = conv_param->output_w_;
  if (dilation_w == 0 || dilation_h == 0 || out_w == 0) {
    return;
  }
  int in_channel = conv_param->input_channel_;
  int in_w = conv_param->input_w_;
  for (int i = 0; i < real_cal_num; i++) {
    int block_start = block_index + i;
    int input_w = block_start % out_w * conv_param->stride_w_ - conv_param->pad_l_;
    int input_h = block_start / out_w * conv_param->stride_h_ - conv_param->pad_u_;
    if (conv_param->input_h_ - input_h < 0 || in_w - input_w < 0) {
      continue;
    }
    int kw_s = MSMAX(0, UP_DIV(-input_w, dilation_w));
    int kw_e = MSMIN(kernel_w, UP_DIV(in_w - input_w, dilation_w));
    int kh_s = MSMAX(0, UP_DIV(-input_h, dilation_h));
    int kh_e = MSMIN(kernel_h, UP_DIV(conv_param->input_h_ - input_h, dilation_h));
    int input_stride = (input_h * in_w + input_w) * in_channel;
    if (dilation_w == 1 && dilation_h == 1) {
      for (int j = kh_s; j < kh_e; j++) {
        int input_plane_offset = (j * kernel_w + kw_s) * in_channel + i * in_channel * kernel_plane;
        int input_y_stride = j * in_w * in_channel + input_stride;
        int input_x_stride = input_y_stride + kw_s * in_channel;
        memcpy(packed_input + input_plane_offset, input_data + input_x_stride,
               (kw_e - kw_s) * in_channel * sizeof(float));
      }  // kernel_h loop
    } else {
      for (int j = kh_s; j < kh_e; j++) {
        int input_y_stride = j * dilation_h * in_w * in_channel + input_stride;
        for (int k = kw_s; k < kw_e; ++k) {
          int input_plane_offset = (j * kernel_w + k) * in_channel + i * in_channel * kernel_plane;
          int input_x_stride = input_y_stride + k * dilation_w * in_channel;
          memcpy(packed_input + input_plane_offset, input_data + input_x_stride, in_channel * sizeof(float));
        }
      }  // kernel_h loop
    }
  }  // tile num loop
}
