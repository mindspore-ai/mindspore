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

#include "nnacl/fp16_grad/convolution_grad_filter.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/errorcode.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

#ifdef ENABLE_NEON

static int FilterGrad32Arm(const float16_t *x, const float16_t *dy, int i_c, int k_idx, float16_t *dw,
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
  for (; i_c < (out_ch & ~31); i_c += 32) {
    float32x4_t sum_0 = vdupq_n_f32(0.0f);
    float32x4_t sum_1 = vdupq_n_f32(0.0f);
    float32x4_t sum_2 = vdupq_n_f32(0.0f);
    float32x4_t sum_3 = vdupq_n_f32(0.0f);
    float32x4_t sum_4 = vdupq_n_f32(0.0f);
    float32x4_t sum_5 = vdupq_n_f32(0.0f);
    float32x4_t sum_6 = vdupq_n_f32(0.0f);
    float32x4_t sum_7 = vdupq_n_f32(0.0f);

    for (int b = 0; b < batch; ++b) {
      const float16_t *x_addr = &x[b * x_size];
      const float16_t *dy_addr = &dy[b * y_size];
      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;

          float16x8_t x_0 = vld1q_f16(x_addr + offset_x);
          float16x8_t dy_0 = vld1q_f16(dy_addr + offset_dy);
          sum_0 = MS_VMLAL_F16(vget_low_f16(x_0), vget_low_f16(dy_0), sum_0);
          sum_1 = MS_VMLAL_F16(vget_high_f16(x_0), vget_high_f16(dy_0), sum_1);

          float16x8_t x_1 = vld1q_f16(x_addr + offset_x + 8);
          float16x8_t dy_1 = vld1q_f16(dy_addr + offset_dy + 8);
          sum_2 = MS_VMLAL_F16(vget_low_f16(x_1), vget_low_f16(dy_1), sum_2);
          sum_3 = MS_VMLAL_F16(vget_high_f16(x_1), vget_high_f16(dy_1), sum_3);

          float16x8_t x_2 = vld1q_f16(x_addr + offset_x + 16);
          float16x8_t dy_2 = vld1q_f16(dy_addr + offset_dy + 16);
          sum_4 = MS_VMLAL_F16(vget_low_f16(x_2), vget_low_f16(dy_2), sum_4);
          sum_5 = MS_VMLAL_F16(vget_high_f16(x_2), vget_high_f16(dy_2), sum_5);

          float16x8_t x_3 = vld1q_f16(x_addr + offset_x + 24);
          float16x8_t dy_3 = vld1q_f16(dy_addr + offset_dy + 24);
          sum_6 = MS_VMLAL_F16(vget_low_f16(x_3), vget_low_f16(dy_3), sum_6);
          sum_7 = MS_VMLAL_F16(vget_high_f16(x_3), vget_high_f16(dy_3), sum_7);
        }
      }
    }
    // store into memory
    for (int l = 0; l < 4; l++) {
      dw[(i_c + l) * k_spatial + k_idx] = sum_0[l];
      dw[(i_c + 4 + l) * k_spatial + k_idx] = sum_1[l];
      dw[(i_c + 8 + l) * k_spatial + k_idx] = sum_2[l];
      dw[(i_c + 12 + l) * k_spatial + k_idx] = sum_3[l];
      dw[(i_c + 16 + l) * k_spatial + k_idx] = sum_4[l];
      dw[(i_c + 20 + l) * k_spatial + k_idx] = sum_5[l];
      dw[(i_c + 24 + l) * k_spatial + k_idx] = sum_6[l];
      dw[(i_c + 28 + l) * k_spatial + k_idx] = sum_7[l];
    }
  }
  return i_c;
}

static int FilterGrad16Arm(const float16_t *x, const float16_t *dy, int i_c, int k_idx, float16_t *dw,
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
    float32x4_t sum_0 = vdupq_n_f32(0.0f);
    float32x4_t sum_1 = vdupq_n_f32(0.0f);
    float32x4_t sum_2 = vdupq_n_f32(0.0f);
    float32x4_t sum_3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < batch; ++b) {
      const float16_t *x_addr = &x[b * x_size];
      const float16_t *dy_addr = &dy[b * y_size];
      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;

          float16x8_t x_0 = vld1q_f16(x_addr + offset_x);
          float16x8_t dy_0 = vld1q_f16(dy_addr + offset_dy);
          sum_0 = MS_VMLAL_F16(vget_low_f16(x_0), vget_low_f16(dy_0), sum_0);
          sum_1 = MS_VMLAL_F16(vget_high_f16(x_0), vget_high_f16(dy_0), sum_1);

          float16x8_t x_1 = vld1q_f16(x_addr + offset_x + 8);
          float16x8_t dy_1 = vld1q_f16(dy_addr + offset_dy + 8);
          sum_2 = MS_VMLAL_F16(vget_low_f16(x_1), vget_low_f16(dy_1), sum_2);
          sum_3 = MS_VMLAL_F16(vget_high_f16(x_1), vget_high_f16(dy_1), sum_3);
        }
      }
    }
    for (int l = 0; l < 4; l++) {
      dw[(i_c + l) * k_spatial + k_idx] = sum_0[l];
      dw[(i_c + l + 4) * k_spatial + k_idx] = sum_1[l];
      dw[(i_c + l + 8) * k_spatial + k_idx] = sum_2[l];
      dw[(i_c + l + 12) * k_spatial + k_idx] = sum_3[l];
    }
  }
  return i_c;
}

static int FilterGrad8Arm(const float16_t *x, const float16_t *dy, int i_c, int k_idx, float16_t *dw,
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
  for (; i_c < (out_ch & ~7); i_c += 8) {
    float32x4_t sum_0 = vdupq_n_f32(0.0f);
    float32x4_t sum_1 = vdupq_n_f32(0.0f);

    for (int b = 0; b < batch; ++b) {
      const float16_t *x_addr = &x[b * x_size];
      const float16_t *dy_addr = &dy[b * y_size];
      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;
          float16x8_t x_0 = vld1q_f16(x_addr + offset_x);
          float16x8_t dy_0 = vld1q_f16(dy_addr + offset_dy);
          sum_0 = MS_VMLAL_F16(vget_low_f16(x_0), vget_low_f16(dy_0), sum_0);
          sum_1 = MS_VMLAL_F16(vget_high_f16(x_0), vget_high_f16(dy_0), sum_1);
        }
      }
    }
    for (int l = 0; l < 4; l++) {
      dw[(i_c + l) * k_spatial + k_idx] = sum_0[l];
      dw[(i_c + 4 + l) * k_spatial + k_idx] = sum_1[l];
    }
  }
  return i_c;
}

static int FilterGrad4Arm(const float16_t *x, const float16_t *dy, int i_c, int k_idx, float16_t *dw,
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
  for (; i_c < (out_ch & ~3); i_c += 4) {
    float32x4_t sum_0 = vdupq_n_f32(0.0f);

    for (int b = 0; b < batch; ++b) {
      const float16_t *x_addr = &x[b * x_size];
      const float16_t *dy_addr = &dy[b * y_size];

      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;
          float16x4_t x_0 = vld1_f16(x_addr + offset_x);
          float16x4_t dy_0 = vld1_f16(dy_addr + offset_dy);
          sum_0 = MS_VMLAL_F16(x_0, dy_0, sum_0);
        }
      }
    }
    dw[(i_c + 0) * k_spatial + k_idx] = sum_0[0];
    dw[(i_c + 1) * k_spatial + k_idx] = sum_0[1];
    dw[(i_c + 2) * k_spatial + k_idx] = sum_0[2];
    dw[(i_c + 3) * k_spatial + k_idx] = sum_0[3];
  }
  return i_c;
}

static int FilterGradLeftoverArm(const float16_t *x, const float16_t *dy, int i_c, int k_idx, float16_t *dw,
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
  int leftover = out_ch - i_c;
  if (leftover > 0) {
    float32x4_t sum_0 = vdupq_n_f32(0.0f);

    for (int b = 0; b < batch; ++b) {
      const float16_t *x_addr = &x[b * x_size];
      const float16_t *dy_addr = &dy[b * y_size];

      for (int i = 0; i < m; i++) {
        int idx = i;
        int input_h = idx / out_w * conv_param->stride_h_;
        int input_w = idx % out_w * conv_param->stride_w_;
        int input_row = -conv_param->pad_u_ + i_kh + input_h;
        int input_col = -conv_param->pad_l_ + i_kw + input_w;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset_x = (input_row * in_w + input_col) * out_ch + i_c;
          int offset_dy = idx * out_ch + i_c;
          float16x4_t x_0 = vld1_f16(x_addr + offset_x);
          float16x4_t dy_0 = vld1_f16(dy_addr + offset_dy);
          sum_0 = MS_VMLAL_F16(x_0, dy_0, sum_0);
        }
      }
    }
    for (int l = 0; l < leftover; l++) {
      dw[(i_c + l) * k_spatial + k_idx] = sum_0[l];
    }
  }
  return out_ch;
}

#endif

int ConvDwFilterFp16Grad(const float16_t *x, const float16_t *dy, float16_t *dw, int start, int count,
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
#ifdef ENABLE_NEON
    i_c = FilterGrad32Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = FilterGrad16Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = FilterGrad8Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = FilterGrad4Arm(x, dy, i_c, k_idx, dw, conv_param);
    i_c = FilterGradLeftoverArm(x, dy, i_c, k_idx, dw, conv_param);
#endif
    for (; i_c < out_ch; i_c++) {
      float sum = 0;
      for (int b = 0; b < batch; ++b) {
        const float16_t *x_addr = &x[b * x_size];
        const float16_t *dy_addr = &dy[b * y_size];

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
  return NNACL_OK;
}
