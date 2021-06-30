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

#include "nnacl/fp16_grad/convolution_grad_input.h"
#include "nnacl/errorcode.h"
#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif

static int ConvDwInputGrad16(const float16_t *dy, const float16_t *w, float16_t *dx, int start, int end,
                             const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_spatial = conv_param->output_h_ * conv_param->output_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int k_spatial = k_h * k_w;
  int batch = conv_param->input_batch_;
  int in_size = in_h * in_w * in_ch;
  int out_size = out_h * out_w * out_ch;

  int j = start;
  for (; j <= (end - C16NUM); j += C16NUM) {
    float16_t *c = dx + j;
    const float16_t *mat_b[C16NUM];
    for (int j_i = 0; j_i < C16NUM; j_i++) {
      mat_b[j_i] = w + (j + j_i) * k_spatial;
    }
    for (int si = 0; si < out_spatial; si++) {
      const float16_t *a = dy + j + si * out_ch;
      int output_row = (si) / out_w;
      int output_col = (si) % out_w;
      int row_stride_offset = -conv_param->pad_u_ + output_row * conv_param->stride_h_;
      int col_stride_offset = -conv_param->pad_l_ + output_col * conv_param->stride_w_;
      for (int k = 0; k < k_spatial; k++) {
        int kernel_row = k / k_w;
        int kernel_col = k % k_w;
        int input_row = kernel_row * conv_param->dilation_h_ + row_stride_offset;
        int input_col = kernel_col * conv_param->dilation_w_ + col_stride_offset;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset = (input_row * in_w + input_col) * in_ch;
#ifdef ENABLE_ARM
#ifdef ENABLE_ARM64
          float16x8_t mat_b0 = {mat_b[0][k], mat_b[1][k], mat_b[2][k], mat_b[3][k],
                                mat_b[4][k], mat_b[5][k], mat_b[6][k], mat_b[7][k]};
          float16x8_t mat_b1 = {mat_b[8][k],  mat_b[9][k],  mat_b[10][k], mat_b[11][k],
                                mat_b[12][k], mat_b[13][k], mat_b[14][k], mat_b[15][k]};
#else
          float16x4_t mat_b00;
          float16x4_t mat_b01;
          float16x4_t mat_b10;
          float16x4_t mat_b11;
          asm volatile(
            "vld1.16 %0[0], [%2]\n"
            "vld1.16 %0[1], [%3]\n"
            "vld1.16 %0[2], [%4]\n"
            "vld1.16 %0[3], [%5]\n"
            "vld1.16 %1[0], [%6]\n"
            "vld1.16 %1[1], [%7]\n"
            "vld1.16 %1[2], [%8]\n"
            "vld1.16 %1[3], [%9]\n"
            : "=w"(mat_b00), "=w"(mat_b01)
            : "r"(mat_b[0] + k), "r"(mat_b[1] + k), "r"(mat_b[2] + k), "r"(mat_b[3] + k), "r"(mat_b[4] + k),
              "r"(mat_b[5] + k), "r"(mat_b[6] + k), "r"(mat_b[7] + k)
            :);
          asm volatile(
            "vld1.16 %0[0], [%2]\n"
            "vld1.16 %0[1], [%3]\n"
            "vld1.16 %0[2], [%4]\n"
            "vld1.16 %0[3], [%5]\n"
            "vld1.16 %1[0], [%6]\n"
            "vld1.16 %1[1], [%7]\n"
            "vld1.16 %1[2], [%8]\n"
            "vld1.16 %1[3], [%9]\n"
            : "=w"(mat_b10), "=w"(mat_b11)
            : "r"(mat_b[8] + k), "r"(mat_b[9] + k), "r"(mat_b[10] + k), "r"(mat_b[11] + k), "r"(mat_b[12] + k),
              "r"(mat_b[13] + k), "r"(mat_b[14] + k), "r"(mat_b[15] + k)
            :);
          float16x8_t mat_b0 = vcombine_f16(mat_b00, mat_b01);
          float16x8_t mat_b1 = vcombine_f16(mat_b10, mat_b11);
#endif
          for (int b = 0; b < batch; b++) {
            int dx_offset = b * in_size + offset;
            int dy_offset = b * out_size;
            float16x8_t mat_c0 = vld1q_f16(c + dx_offset);
            float16x8_t mat_a0 = vld1q_f16(a + dy_offset);
            mat_c0 = vfmaq_f16(mat_c0, mat_b0, mat_a0);
            vst1q_f16(c + dx_offset, mat_c0);

            float16x8_t mat_c1 = vld1q_f16(c + dx_offset + 8);
            float16x8_t mat_a1 = vld1q_f16(a + dy_offset + 8);
            mat_c1 = vfmaq_f16(mat_c1, mat_b1, mat_a1);
            vst1q_f16(c + dx_offset + 8, mat_c1);
          }
#else
          for (int b = 0; b < batch; b++) {
            int dx_offset = b * in_size + offset;
            int dy_offset = b * out_size;
            for (int j_i = 0; j_i < C16NUM; j_i++) {
              c[dx_offset + j_i] += a[dy_offset + j_i] * mat_b[j_i][k];
            }
          }
#endif
        }
      }
    }
  }
  return j;
}

static int ConvDwInputGrad8(const float16_t *dy, const float16_t *w, float16_t *dx, int start, int end,
                            const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_spatial = conv_param->output_h_ * conv_param->output_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int k_spatial = k_h * k_w;
  int batch = conv_param->input_batch_;
  int in_size = in_h * in_w * in_ch;
  int out_size = out_h * out_w * out_ch;

  int j = start;
  for (; j <= (end - C8NUM); j += C8NUM) {
    float16_t *c = dx + j;
    const float16_t *mat_b[C8NUM];
    for (int j_i = 0; j_i < C8NUM; j_i++) {
      mat_b[j_i] = w + (j + j_i) * k_spatial;
    }

    for (int si = 0; si < out_spatial; si++) {
      const float16_t *a = dy + j + si * out_ch;
      int output_row = (si) / out_w;
      int output_col = (si) % out_w;
      int row_stride_offset = -conv_param->pad_u_ + output_row * conv_param->stride_h_;
      int col_stride_offset = -conv_param->pad_l_ + output_col * conv_param->stride_w_;
      for (int k = 0; k < k_spatial; k++) {
        int kernel_row = k / k_w;
        int kernel_col = k % k_w;
        int input_row = kernel_row * conv_param->dilation_h_ + row_stride_offset;
        int input_col = kernel_col * conv_param->dilation_w_ + col_stride_offset;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset = (input_row * in_w + input_col) * in_ch;
#ifdef ENABLE_ARM
#ifdef ENABLE_ARM64
          float16x8_t mat_b0 = {mat_b[0][k], mat_b[1][k], mat_b[2][k], mat_b[3][k],
                                mat_b[4][k], mat_b[5][k], mat_b[6][k], mat_b[7][k]};
#else
          float16x4_t mat_b00;
          float16x4_t mat_b01;
          asm volatile(
            "vld1.16 %0[0], [%2]\n"
            "vld1.16 %0[1], [%3]\n"
            "vld1.16 %0[2], [%4]\n"
            "vld1.16 %0[3], [%5]\n"
            "vld1.16 %1[0], [%6]\n"
            "vld1.16 %1[1], [%7]\n"
            "vld1.16 %1[2], [%8]\n"
            "vld1.16 %1[3], [%9]\n"
            : "=w"(mat_b00), "=w"(mat_b01)
            : "r"(mat_b[0] + k), "r"(mat_b[1] + k), "r"(mat_b[2] + k), "r"(mat_b[3] + k), "r"(mat_b[4] + k),
              "r"(mat_b[5] + k), "r"(mat_b[6] + k), "r"(mat_b[7] + k)
            :);
          float16x8_t mat_b0 = vcombine_f16(mat_b00, mat_b01);
#endif
          for (int b = 0; b < batch; b++) {
            int dx_offset = b * in_size + offset;
            int dy_offset = b * out_size;
            float16x8_t mat_c0 = vld1q_f16(c + dx_offset);
            float16x8_t mat_a0 = vld1q_f16(a + dy_offset);
            mat_c0 = vfmaq_f16(mat_c0, mat_b0, mat_a0);
            vst1q_f16(c + dx_offset, mat_c0);
          }
#else
          for (int b = 0; b < batch; b++) {
            int dx_offset = b * in_size + offset;
            int dy_offset = b * out_size;
            for (int j_i = 0; j_i < C8NUM; j_i++) {
              c[dx_offset + j_i] += a[dy_offset + j_i] * mat_b[j_i][k];
            }
          }
#endif
        }
      }
    }
  }
  return j;
}

static int ConvDwInputGrad4(const float16_t *dy, const float16_t *w, float16_t *dx, int start, int end,
                            const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_spatial = conv_param->output_h_ * conv_param->output_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int k_spatial = k_h * k_w;
  int batch = conv_param->input_batch_;
  int in_size = in_h * in_w * in_ch;
  int out_size = out_h * out_w * out_ch;

  int j = start;
  for (; j <= (end - C4NUM); j += C4NUM) {
    float16_t *c = dx + j;
    const float16_t *mat_b_0 = w + (j + 0) * k_spatial;
    const float16_t *mat_b_1 = w + (j + 1) * k_spatial;
    const float16_t *mat_b_2 = w + (j + 2) * k_spatial;
    const float16_t *mat_b_3 = w + (j + 3) * k_spatial;

    for (int si = 0; si < out_spatial; si++) {
      const float16_t *a = dy + j + si * out_ch;
      int output_row = (si) / out_w;
      int output_col = (si) % out_w;
      int row_stride_offset = -conv_param->pad_u_ + output_row * conv_param->stride_h_;
      int col_stride_offset = -conv_param->pad_l_ + output_col * conv_param->stride_w_;
      for (int k = 0; k < k_spatial; k++) {
        int kernel_row = k / k_w;
        int kernel_col = k % k_w;
        int input_row = kernel_row * conv_param->dilation_h_ + row_stride_offset;
        int input_col = kernel_col * conv_param->dilation_w_ + col_stride_offset;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset = (input_row * in_w + input_col) * in_ch;
#ifdef ENABLE_ARM
#ifdef ENABLE_ARM64
          float16x4_t mat_b = {mat_b_0[k], mat_b_1[k], mat_b_2[k], mat_b_3[k]};
#else
          float16x4_t mat_b;
          asm volatile(
            "vld1.16 %0[0], [%1]\n"
            "vld1.16 %0[1], [%2]\n"
            "vld1.16 %0[2], [%3]\n"
            "vld1.16 %0[3], [%4]\n"
            : "=w"(mat_b)
            : "r"(mat_b_0 + k), "r"(mat_b_1 + k), "r"(mat_b_2 + k), "r"(mat_b_3 + k)
            :);
#endif
          for (int b = 0; b < batch; b++) {
            int dx_offset = b * in_size + offset;
            int dy_offset = b * out_size;
            float16x4_t mat_c = vld1_f16(c + dx_offset);
            float16x4_t mat_a = vld1_f16(a + dy_offset);
            mat_c = vfma_f16(mat_c, mat_b, mat_a);
            vst1_f16(c + dx_offset, mat_c);
          }
#else
          for (int b = 0; b < batch; b++) {
            int dx_offset = b * in_size + offset;
            int dy_offset = b * out_size;
            c[dx_offset + 0] += a[dy_offset + 0] * mat_b_0[k];
            c[dx_offset + 1] += a[dy_offset + 1] * mat_b_1[k];
            c[dx_offset + 2] += a[dy_offset + 2] * mat_b_2[k];
            c[dx_offset + 3] += a[dy_offset + 3] * mat_b_3[k];
          }
#endif
        }
      }
    }
  }
  return j;
}

int ConvDwInputGradFp16(const float16_t *dy, const float16_t *w, float16_t *dx, int start, int count,
                        const ConvParameter *conv_param) {
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_w = conv_param->output_w_;
  int out_h = conv_param->output_h_;
  int out_ch = conv_param->output_channel_;
  int in_ch = conv_param->input_channel_;
  int out_spatial = conv_param->output_h_ * conv_param->output_w_;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int k_spatial = k_h * k_w;
  int end = start + count;
  int batch = conv_param->input_batch_;
  int in_size = in_h * in_w * in_ch;
  int out_size = out_h * out_w * out_ch;

  int j = start;
  j = ConvDwInputGrad16(dy, w, dx, j, end, conv_param);
  j = ConvDwInputGrad8(dy, w, dx, j, end, conv_param);
  j = ConvDwInputGrad4(dy, w, dx, j, end, conv_param);
  for (; j < end; j++) {
    float16_t *c = dx + j;
    const float16_t *b = w + j * k_spatial;
    for (int si = 0; si < out_spatial; si++) {
      const float16_t *a = dy + j + si * out_ch;
      int output_row = si / out_w;
      int output_col = si % out_w;
      int row_stride_offset = -conv_param->pad_u_ + output_row * conv_param->stride_h_;
      int col_stride_offset = -conv_param->pad_l_ + output_col * conv_param->stride_w_;
      for (int k = 0; k < k_spatial; k++) {
        int kernel_row = k / k_w;
        int kernel_col = k % k_w;
        int input_row = kernel_row * conv_param->dilation_h_ + row_stride_offset;
        int input_col = kernel_col * conv_param->dilation_w_ + col_stride_offset;
        if (((unsigned)(input_row) < (unsigned)(in_h)) && ((unsigned)(input_col) < (unsigned)(in_w))) {
          int offset = (input_row * in_w + input_col) * in_ch;
          for (int bi = 0; bi < batch; bi++) {
            c[bi * in_size + offset + 0] += a[0 + bi * out_size] * b[k];
          }
        }
      }
    }
  }
  return NNACL_OK;
}
