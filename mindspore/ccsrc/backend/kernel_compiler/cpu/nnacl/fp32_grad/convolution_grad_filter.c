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

#include "nnacl/fp32_grad/convolution_grad_filter.h"
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif

#ifdef ENABLE_ARM
static int FilterGrad16Arm(const float *x, const float *dy, int i_c, int k_idx, float *dw,
                           const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int x_size = in_h * in_w * in_ch;
  int y_size = out_ch * out_h * out_w;
  int k_spatial = k_w * k_h;
  int i_kh = k_idx / k_w;
  int i_kw = k_idx % k_w;
  for (; i_c < (out_ch & ~15); i_c += 16) {
    float32x4_t sum_03_4 = vdupq_n_f32(0.0f);
    float32x4_t sum_47_4 = vdupq_n_f32(0.0f);
    float32x4_t sum_9x_4 = vdupq_n_f32(0.0f);
    float32x4_t sum_12x_4 = vdupq_n_f32(0.0f);
    for (int b = 0; b < batch; ++b) {
      const float *x_addr = &x[b * x_size];
      const float *dy_addr = &dy[b * y_size];
      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;

          float32x4_t x_03_4 = vld1q_f32(x_addr + offset_x);
          float32x4_t dy_03_4 = vld1q_f32(dy_addr + offset_dy);
          sum_03_4 = vmlaq_f32(sum_03_4, x_03_4, dy_03_4);

          float32x4_t x_47_4 = vld1q_f32(x_addr + offset_x + 4);
          float32x4_t dy_47_4 = vld1q_f32(dy_addr + offset_dy + 4);
          sum_47_4 = vmlaq_f32(sum_47_4, x_47_4, dy_47_4);

          float32x4_t x_9x_4 = vld1q_f32(x_addr + offset_x + 8);
          float32x4_t dy_9x_4 = vld1q_f32(dy_addr + offset_dy + 8);
          sum_9x_4 = vmlaq_f32(sum_9x_4, x_9x_4, dy_9x_4);

          float32x4_t x_12x_4 = vld1q_f32(x_addr + offset_x + 12);
          float32x4_t dy_12x_4 = vld1q_f32(dy_addr + offset_dy + 12);
          sum_12x_4 = vmlaq_f32(sum_12x_4, x_12x_4, dy_12x_4);
        }
      }
    }
    dw[(i_c + 0) * k_spatial + k_idx] = sum_03_4[0];
    dw[(i_c + 1) * k_spatial + k_idx] = sum_03_4[1];
    dw[(i_c + 2) * k_spatial + k_idx] = sum_03_4[2];
    dw[(i_c + 3) * k_spatial + k_idx] = sum_03_4[3];

    dw[(i_c + 4) * k_spatial + k_idx] = sum_47_4[0];
    dw[(i_c + 5) * k_spatial + k_idx] = sum_47_4[1];
    dw[(i_c + 6) * k_spatial + k_idx] = sum_47_4[2];
    dw[(i_c + 7) * k_spatial + k_idx] = sum_47_4[3];

    dw[(i_c + 8) * k_spatial + k_idx] = sum_9x_4[0];
    dw[(i_c + 9) * k_spatial + k_idx] = sum_9x_4[1];
    dw[(i_c + 10) * k_spatial + k_idx] = sum_9x_4[2];
    dw[(i_c + 11) * k_spatial + k_idx] = sum_9x_4[3];

    dw[(i_c + 12) * k_spatial + k_idx] = sum_12x_4[0];
    dw[(i_c + 13) * k_spatial + k_idx] = sum_12x_4[1];
    dw[(i_c + 14) * k_spatial + k_idx] = sum_12x_4[2];
    dw[(i_c + 15) * k_spatial + k_idx] = sum_12x_4[3];
  }
  return i_c;
}

static int FilterGrad12Arm(const float *x, const float *dy, int i_c, int k_idx, float *dw,
                           const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int x_size = in_h * in_w * in_ch;
  int y_size = out_ch * out_h * out_w;
  int k_spatial = k_w * k_h;
  int i_kh = k_idx / k_w;
  int i_kw = k_idx % k_w;
  if ((out_ch - i_c) >= 12) {
    float32x4_t sum_03_4 = vdupq_n_f32(0.0f);
    float32x4_t sum_47_4 = vdupq_n_f32(0.0f);
    float32x4_t sum_9x_4 = vdupq_n_f32(0.0f);
    for (int b = 0; b < batch; ++b) {
      const float *x_addr = &x[b * x_size];
      const float *dy_addr = &dy[b * y_size];

      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;

          float32x4_t x_03_4 = vld1q_f32(x_addr + offset_x);
          float32x4_t dy_03_4 = vld1q_f32(dy_addr + offset_dy);
          sum_03_4 = vmlaq_f32(sum_03_4, x_03_4, dy_03_4);

          float32x4_t x_47_4 = vld1q_f32(x_addr + offset_x + 4);
          float32x4_t dy_47_4 = vld1q_f32(dy_addr + offset_dy + 4);
          sum_47_4 = vmlaq_f32(sum_47_4, x_47_4, dy_47_4);

          float32x4_t x_9x_4 = vld1q_f32(x_addr + offset_x + 8);
          float32x4_t dy_9x_4 = vld1q_f32(dy_addr + offset_dy + 8);
          sum_9x_4 = vmlaq_f32(sum_9x_4, x_9x_4, dy_9x_4);
        }
      }
    }
    dw[(i_c + 0) * k_spatial + k_idx] = sum_03_4[0];
    dw[(i_c + 1) * k_spatial + k_idx] = sum_03_4[1];
    dw[(i_c + 2) * k_spatial + k_idx] = sum_03_4[2];
    dw[(i_c + 3) * k_spatial + k_idx] = sum_03_4[3];

    dw[(i_c + 4) * k_spatial + k_idx] = sum_47_4[0];
    dw[(i_c + 5) * k_spatial + k_idx] = sum_47_4[1];
    dw[(i_c + 6) * k_spatial + k_idx] = sum_47_4[2];
    dw[(i_c + 7) * k_spatial + k_idx] = sum_47_4[3];

    dw[(i_c + 8) * k_spatial + k_idx] = sum_9x_4[0];
    dw[(i_c + 9) * k_spatial + k_idx] = sum_9x_4[1];
    dw[(i_c + 10) * k_spatial + k_idx] = sum_9x_4[2];
    dw[(i_c + 11) * k_spatial + k_idx] = sum_9x_4[3];

    i_c += 12;
  }
  return i_c;
}

static int FilterGrad8Arm(const float *x, const float *dy, int i_c, int k_idx, float *dw,
                          const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int x_size = in_h * in_w * in_ch;
  int y_size = out_ch * out_h * out_w;
  int k_spatial = k_w * k_h;
  int i_kh = k_idx / k_w;
  int i_kw = k_idx % k_w;

  if ((out_ch - i_c) >= 8) {
    float32x4_t sum_03_4 = vdupq_n_f32(0.0f);
    float32x4_t sum_47_4 = vdupq_n_f32(0.0f);
    for (int b = 0; b < batch; ++b) {
      const float *x_addr = &x[b * x_size];
      const float *dy_addr = &dy[b * y_size];

      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;

          float32x4_t x_03_4 = vld1q_f32(x_addr + offset_x);
          float32x4_t dy_03_4 = vld1q_f32(dy_addr + offset_dy);
          sum_03_4 = vmlaq_f32(sum_03_4, x_03_4, dy_03_4);

          float32x4_t x_47_4 = vld1q_f32(x_addr + offset_x + 4);
          float32x4_t dy_47_4 = vld1q_f32(dy_addr + offset_dy + 4);
          sum_47_4 = vmlaq_f32(sum_47_4, x_47_4, dy_47_4);
        }
      }
    }
    dw[(i_c + 0) * k_spatial + k_idx] = sum_03_4[0];
    dw[(i_c + 1) * k_spatial + k_idx] = sum_03_4[1];
    dw[(i_c + 2) * k_spatial + k_idx] = sum_03_4[2];
    dw[(i_c + 3) * k_spatial + k_idx] = sum_03_4[3];

    dw[(i_c + 4) * k_spatial + k_idx] = sum_47_4[0];
    dw[(i_c + 5) * k_spatial + k_idx] = sum_47_4[1];
    dw[(i_c + 6) * k_spatial + k_idx] = sum_47_4[2];
    dw[(i_c + 7) * k_spatial + k_idx] = sum_47_4[3];
    i_c += 8;
  }
  return i_c;
}
static int FilterGrad4Arm(const float *x, const float *dy, int i_c, int k_idx, float *dw,
                          const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int x_size = in_h * in_w * in_ch;
  int y_size = out_ch * out_h * out_w;
  int k_spatial = k_w * k_h;
  int i_kh = k_idx / k_w;
  int i_kw = k_idx % k_w;
  if ((out_ch - i_c) >= 4) {
    float32x4_t sum_4 = vdupq_n_f32(0.0f);

    for (int b = 0; b < batch; ++b) {
      const float *x_addr = &x[b * x_size];
      const float *dy_addr = &dy[b * y_size];

      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;

          float32x4_t x_4 = vld1q_f32(x_addr + offset_x);
          float32x4_t dy_4 = vld1q_f32(dy_addr + offset_dy);
          sum_4 = vmlaq_f32(sum_4, x_4, dy_4);
        }
      }
    }
    dw[(i_c + 0) * k_spatial + k_idx] = sum_4[0];
    dw[(i_c + 1) * k_spatial + k_idx] = sum_4[1];
    dw[(i_c + 2) * k_spatial + k_idx] = sum_4[2];
    dw[(i_c + 3) * k_spatial + k_idx] = sum_4[3];
    i_c += 4;
  }
  return i_c;
}

static int Filtergrad2Arm(const float *x, const float *dy, int i_c, int k_idx, float *dw,
                          const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int x_size = in_h * in_w * in_ch;
  int y_size = out_ch * out_h * out_w;
  int k_spatial = k_w * k_h;
  int i_kh = k_idx / k_w;
  int i_kw = k_idx % k_w;

  if ((out_ch - i_c) >= 2) {
    float32x2_t sum_2 = vdup_n_f32(0.0f);
    for (int b = 0; b < batch; ++b) {
      const float *x_addr = &x[b * x_size];
      const float *dy_addr = &dy[b * y_size];

      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;

          float32x2_t x_4 = vld1_f32(x_addr + offset_x);
          float32x2_t dy_4 = vld1_f32(dy_addr + offset_dy);
          sum_2 = vmla_f32(sum_2, x_4, dy_4);
        }
      }
    }
    dw[(i_c + 0) * k_spatial + k_idx] = sum_2[0];
    dw[(i_c + 1) * k_spatial + k_idx] = sum_2[1];
    i_c += 2;
  }
  return i_c += 2;
}
#endif
int ConvDwFilterGrad(const float *x, const float *dy, float *dw, int start, int count,
                     const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int batch = conv_param->output_batch_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int m = out_h * out_w;
  int x_size = in_h * in_w * in_ch;
  int y_size = out_ch * out_h * out_w;
  int k_spatial = k_w * k_h;

  for (int i_k = 0; i_k < count; i_k++) {
    int k_idx = start + i_k;
    int i_kh = k_idx / k_w;
    int i_kw = k_idx % k_w;
    int i_c = 0;
#ifdef ENABLE_ARM
    i_c = FilterGrad16Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = FilterGrad12Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = FilterGrad8Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = FilterGrad4Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = Filtergrad2Arm(x, dy, i_c, k_idx, dw, conv_param);
#endif
    for (; i_c < out_ch; i_c++) {
      float sum = 0;
      for (int b = 0; b < batch; ++b) {
        const float *x_addr = &x[b * x_size];
        const float *dy_addr = &dy[b * y_size];

        for (int i = 0; i < m; i++) {
          int idx = i;
          int input_h = idx / out_w * conv_param->stride_h_;
          int input_w = idx % out_w * conv_param->stride_w_;
          int input_row = -conv_param->pad_u_ + i_kh + input_h;
          int input_col = -conv_param->pad_l_ + i_kw + input_w;
          if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
            int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
            int offset_dy = idx * out_ch + i_c;
            sum += x_addr[offset_x] * dy_addr[offset_dy];
          }
        }
      }
      dw[i_c * k_spatial + k_idx] = sum;
    }
  }
  return 0;
}
