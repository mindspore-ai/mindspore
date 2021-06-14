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

#include "nnacl/fp32_grad/convolution_grad_input.h"
#include "nnacl/errorcode.h"
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif

int ConvDwInputGrad(const float *dy, const float *w, float *dx, int start, int count, const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_spatial = conv_param->output_h_ * conv_param->output_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int k_spatial = k_h * k_w;
  int end = start + count;

  int j = start;
  for (; j <= (end - C4NUM); j += C4NUM) {
    float *c = dx + j;
    const float *mat_b_0 = w + (j + 0) * k_spatial;
    const float *mat_b_1 = w + (j + 1) * k_spatial;
    const float *mat_b_2 = w + (j + 2) * k_spatial;
    const float *mat_b_3 = w + (j + 3) * k_spatial;

    for (int si = 0; si < out_spatial; si++) {
      const float *a = dy + j + si * out_ch;
#ifdef ENABLE_ARM
      float32x4_t mat_a = vld1q_f32(a);
#else
      float mat_a[4] = {a[0], a[1], a[2], a[3]};
#endif
      int output_row = (si) / out_w;
      int output_col = (si) % out_w;
      for (int k = 0; k < k_spatial; k++) {
        int row_stride_offset = output_row * conv_param->stride_h_;
        int col_stride_offset = output_col * conv_param->stride_w_;
        int kernel_row = k / k_w;
        int kernel_col = k % k_w;
        int input_row = -conv_param->pad_u_ + kernel_row * conv_param->dilation_h_ + row_stride_offset;
        int input_col = -conv_param->pad_l_ + kernel_col * conv_param->dilation_w_ + col_stride_offset;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset = (input_row * in_w + input_col) * in_ch;
#ifdef ENABLE_ARM
          float32x4_t mat_b = {mat_b_0[k], mat_b_1[k], mat_b_2[k], mat_b_3[k]};
          float32x4_t mat_c = vld1q_f32(c + offset);
          mat_c = vmlaq_f32(mat_c, mat_b, mat_a);
          vst1q_f32(c + offset, mat_c);
#else
          c[offset + 0] += mat_a[0] * mat_b_0[k];
          c[offset + 1] += mat_a[1] * mat_b_1[k];
          c[offset + 2] += mat_a[2] * mat_b_2[k];
          c[offset + 3] += mat_a[3] * mat_b_3[k];
#endif
        }
      }
    }
  }

  for (; j < end; j++) {
    float *c = dx + j;
    const float *b = w + j * k_spatial;
    for (int si = 0; si < out_spatial; si++) {
      const float *a = dy + j + si * out_ch;
      int output_row = si / out_w;
      int output_col = si % out_w;
      int row_stride_offset = output_row * conv_param->stride_h_;
      int col_stride_offset = output_col * conv_param->stride_w_;
      for (int k = 0; k < k_spatial; k++) {
        int kernel_row = k / k_w;
        int kernel_col = k % k_w;
        int input_row = -conv_param->pad_u_ + kernel_row * conv_param->dilation_h_ + row_stride_offset;
        int input_col = -conv_param->pad_l_ + kernel_col * conv_param->dilation_w_ + col_stride_offset;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset = (input_row * in_w + input_col) * in_ch;
          c[offset] += a[0] * b[k];
        }
      }
    }
  }
  return NNACL_OK;
}
