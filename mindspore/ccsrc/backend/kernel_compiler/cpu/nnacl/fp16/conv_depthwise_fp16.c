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

#include "nnacl/fp16/conv_depthwise_fp16.h"
#include <string.h>
#include "nnacl/fp16/activation_fp16.h"

#ifdef ENABLE_ARM82_A32
void ConvDwFp16Row(float16_t *output_ptr, const float16_t *input_ptr, const float16_t *weight_ptr, size_t num_pixels,
                   size_t output_channel, size_t input_step) {
  for (int i = 0; i < num_pixels; i++) {
    for (int c = 0; c < output_channel; c++) {
      *output_ptr++ += weight_ptr[c] * input_ptr[c];
    }
    input_ptr += input_step;
  }
}
#endif

#ifdef ENABLE_ARM
static void ConvDw3x3RowLeftFp16(const float16_t *src, float16_t *line, int lw, int channel) {
  MS_FLOAT16X8 v0, v1, v2, v3;
  v0 = MS_MOVQ_F16((float16_t)0.0);
  int ic = 0;
  for (; ic < channel - 7; ic += 8) {
    v1 = MS_LDQ_F16(src + ic);
    v2 = MS_LDQ_F16(src + channel + ic);
    v3 = MS_LDQ_F16(src + 2 * channel + ic);
    MS_FLOAT16X8 b0 = MS_SUBQ_F16(v0, v2);
    MS_FLOAT16X8 b1 = MS_ADDQ_F16(v1, v2);
    MS_FLOAT16X8 b2 = MS_SUBQ_F16(v2, v1);
    MS_FLOAT16X8 b3 = MS_SUBQ_F16(v3, v1);
    MS_STQ_F16(line + lw * ic, b0);
    MS_STQ_F16(line + lw * ic + 8, b1);
    MS_STQ_F16(line + lw * ic + 16, b2);
    MS_STQ_F16(line + lw * ic + 24, b3);
  }
  if (ic < channel) {
    float16_t *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 16, 0, 16);
    memset(remain_line + 24, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float16_t d1 = src[i + ic];
      float16_t d2 = src[i + ic + channel];
      float16_t d3 = src[i + ic + 2 * channel];
      remain_line[i] = (float16_t)0.0 - d2;
      remain_line[i + 8] = d1 + d2;
      remain_line[i + 16] = d2 - d1;
      remain_line[i + 24] = d3 - d1;
    }
  }
}

static void ConvDw3x3RowMiddleFp16(const float16_t *src, float16_t *line, int lw, int channel) {
  MS_FLOAT16X8 v0, v1, v2, v3;
  int ic = 0;
  for (; ic < channel - 7; ic += 8) {
    v0 = MS_LDQ_F16(src + ic);
    v1 = MS_LDQ_F16(src + channel + ic);
    v2 = MS_LDQ_F16(src + 2 * channel + ic);
    v3 = MS_LDQ_F16(src + 3 * channel + ic);
    MS_FLOAT16X8 b0 = MS_SUBQ_F16(v0, v2);
    MS_FLOAT16X8 b1 = MS_ADDQ_F16(v1, v2);
    MS_FLOAT16X8 b2 = MS_SUBQ_F16(v2, v1);
    MS_FLOAT16X8 b3 = MS_SUBQ_F16(v3, v1);
    MS_STQ_F16(line + lw * ic, b0);
    MS_STQ_F16(line + lw * ic + 8, b1);
    MS_STQ_F16(line + lw * ic + 16, b2);
    MS_STQ_F16(line + lw * ic + 24, b3);
  }
  if (ic < channel) {
    float16_t *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 16, 0, 16);
    memset(remain_line + 24, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float16_t d0 = src[i + ic];
      float16_t d1 = src[i + ic + channel];
      float16_t d2 = src[i + ic + 2 * channel];
      float16_t d3 = src[i + ic + 3 * channel];
      remain_line[i] = d0 - d2;
      remain_line[i + 8] = d1 + d2;
      remain_line[i + 16] = d2 - d1;
      remain_line[i + 24] = d3 - d1;
    }
  }
}

static void ConvDw3x3RowRightFp16(const float16_t *src, float16_t *line, int lw, int channel) {
  MS_FLOAT16X8 v0, v1, v2, v3;
  int ic = 0;
  v3 = MS_MOVQ_F16((float16_t)0.0);
  for (; ic < channel - 7; ic += 8) {
    v0 = MS_LDQ_F16(src + ic);
    v1 = MS_LDQ_F16(src + channel + ic);
    v2 = MS_LDQ_F16(src + 2 * channel + ic);
    MS_FLOAT16X8 b0 = MS_SUBQ_F16(v0, v2);
    MS_FLOAT16X8 b1 = MS_ADDQ_F16(v1, v2);
    MS_FLOAT16X8 b2 = MS_SUBQ_F16(v2, v1);
    MS_FLOAT16X8 b3 = MS_SUBQ_F16(v3, v1);
    MS_STQ_F16(line + lw * ic, b0);
    MS_STQ_F16(line + lw * ic + 8, b1);
    MS_STQ_F16(line + lw * ic + 16, b2);
    MS_STQ_F16(line + lw * ic + 24, b3);
  }
  if (ic < channel) {
    float16_t *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 16, 0, 16);
    memset(remain_line + 24, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float16_t d0 = src[i + ic];
      float16_t d1 = src[i + ic + channel];
      float16_t d2 = src[i + ic + 2 * channel];
      remain_line[i] = d0 - d2;
      remain_line[i + 8] = d1 + d2;
      remain_line[i + 16] = d2 - d1;
      remain_line[i + 24] = (float16_t)0.0 - d1;
    }
  }
}

static void ConvDw3x3RowSingleFp16(const float16_t *src, float16_t *line, int lw, int channel) {
  MS_FLOAT16X8 v0, v1, v2;
  int ic = 0;
  v2 = MS_MOVQ_F16((float16_t)0.0);
  for (; ic < channel - 7; ic += 8) {
    v0 = MS_LDQ_F16(src + ic);
    v1 = MS_LDQ_F16(src + channel + ic);
    MS_FLOAT16X8 b2 = MS_SUBQ_F16(v2, v1);
    MS_STQ_F16(line + lw * ic, v0);
    MS_STQ_F16(line + lw * ic + 8, v1);
    MS_STQ_F16(line + lw * ic + 16, b2);
    memset(line + lw * ic + 24, 0, 16);
  }
  if (ic < channel) {
    float16_t *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 16, 0, 16);
    memset(remain_line + 24, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float16_t d0 = src[i + ic];
      float16_t d1 = src[i + ic + channel];
      remain_line[i] = d0;
      remain_line[i + 8] = d1;
      remain_line[i + 16] = (float16_t)0.0 - d1;
    }
  }
}

static void ConvDw3x3InitTopFp16(const float16_t *src, float16_t **lines, int width, int channel) {
  float16_t *line0 = lines[0];
  float16_t *line1 = lines[1];
  float16_t *line2 = lines[2];
  int c8 = UP_ROUND(channel, C8NUM);
  int lw = UP_DIV(width, C2NUM) * C4NUM;
  memset(line0, 0, c8 * lw * sizeof(float16_t));
  ConvDw3x3RowLeftFp16(src, line1, lw, channel);
  ConvDw3x3RowLeftFp16(src + width * channel, line2, lw, channel);
  int ow = 2;
  for (; ow < width - 2; ow += 2) {
    ConvDw3x3RowMiddleFp16(src + (ow - 1) * channel, line1 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowMiddleFp16(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 8, lw, channel);
  }
  int remain = width - ow;
  if (remain == 2) {
    ConvDw3x3RowRightFp16(src + (ow - 1) * channel, line1 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowRightFp16(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 8, lw, channel);
  } else if (remain == 1) {
    ConvDw3x3RowSingleFp16(src + (ow - 1) * channel, line1 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowSingleFp16(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 8, lw, channel);
  }
}

static void ConvDw3x3InitRowFp16(const float16_t *src, float16_t **lines, int width, int channel) {
  float16_t *line0 = lines[0];
  float16_t *line1 = lines[1];
  float16_t *line2 = lines[2];
  int lw = UP_DIV(width, C2NUM) * C4NUM;
  ConvDw3x3RowLeftFp16(src - width * channel, line0, lw, channel);
  ConvDw3x3RowLeftFp16(src, line1, lw, channel);
  ConvDw3x3RowLeftFp16(src + width * channel, line2, lw, channel);
  int ow = 2;
  for (; ow < width - 2; ow += 2) {
    ConvDw3x3RowMiddleFp16(src - width * channel + (ow - 1) * channel, line0 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowMiddleFp16(src + (ow - 1) * channel, line1 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowMiddleFp16(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 8, lw, channel);
  }
  int remain = width - ow;
  if (remain == 2) {
    ConvDw3x3RowRightFp16(src - width * channel + (ow - 1) * channel, line0 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowRightFp16(src + (ow - 1) * channel, line1 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowRightFp16(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 8, lw, channel);
  } else if (remain == 1) {
    ConvDw3x3RowSingleFp16(src - width * channel + (ow - 1) * channel, line0 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowSingleFp16(src + (ow - 1) * channel, line1 + 2 * ow * 8, lw, channel);
    ConvDw3x3RowSingleFp16(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 8, lw, channel);
  }
}

static void ConvDw3x3RowFp16(const float16_t *src, float16_t **lines, int width, int channel) {
  float16_t *tmp = lines[0];
  lines[0] = lines[1];
  lines[1] = lines[2];
  lines[2] = tmp;
  int c8 = UP_ROUND(channel, C8NUM);
  int lw = UP_DIV(width, C2NUM) * C4NUM;
  memset(tmp, 0, c8 * lw * sizeof(float16_t));
  ConvDw3x3RowLeftFp16(src, tmp, lw, channel);
  int ow = 2;
  for (; ow < width - 2; ow += 2) {
    ConvDw3x3RowMiddleFp16(src + (ow - 1) * channel, tmp + 2 * ow * 8, lw, channel);
  }
  int remain = width - ow;
  if (remain == 2) {
    ConvDw3x3RowRightFp16(src + (ow - 1) * channel, tmp + 2 * ow * 8, lw, channel);
  } else if (remain == 1) {
    ConvDw3x3RowSingleFp16(src + (ow - 1) * channel, tmp + 2 * ow * 8, lw, channel);
  }
}

static void ConvDw3x3BottomFp16(float16_t **lines, int width, int channel) {
  float16_t *tmp = lines[0];
  lines[0] = lines[1];
  lines[1] = lines[2];
  lines[2] = tmp;
  int c8 = UP_ROUND(channel, C8NUM);
  memset(tmp, 0, UP_DIV(width, C2NUM) * c8 * C4NUM * sizeof(float16_t));
}

void ConvDw3x3LineFp16(float16_t *dst, float16_t **lines, const float16_t *weight, const float16_t *bias_data,
                       int width, int ori_channel, bool relu, bool relu6) {
  int channel = ori_channel;
  float16_t *line0 = lines[0];
  float16_t *line1 = lines[1];
  float16_t *line2 = lines[2];
  for (; channel > 0; channel -= 8) {
    MS_FLOAT16X8 bias = MS_LDQ_F16(bias_data);
    bias_data += 8;
    MS_FLOAT16X8 g00 = MS_LDQ_F16(weight);
    MS_FLOAT16X8 g01 = MS_LDQ_F16(weight + 8);
    MS_FLOAT16X8 g02 = MS_LDQ_F16(weight + 16);
    MS_FLOAT16X8 g03 = MS_LDQ_F16(weight + 24);
    MS_FLOAT16X8 g10 = MS_LDQ_F16(weight + 32);
    MS_FLOAT16X8 g11 = MS_LDQ_F16(weight + 40);
    MS_FLOAT16X8 g12 = MS_LDQ_F16(weight + 48);
    MS_FLOAT16X8 g13 = MS_LDQ_F16(weight + 56);
    MS_FLOAT16X8 g20 = MS_LDQ_F16(weight + 64);
    MS_FLOAT16X8 g21 = MS_LDQ_F16(weight + 72);
    MS_FLOAT16X8 g22 = MS_LDQ_F16(weight + 80);
    MS_FLOAT16X8 g23 = MS_LDQ_F16(weight + 88);
    weight += 96;
    float16_t *cur_dst = dst;
    int ow = 0;
    for (; ow < width - 1; ow += 2) {
      MS_FLOAT16X8 acc0 = MS_MULQ_F16(MS_LDQ_F16(line0), g00);
      MS_FLOAT16X8 acc1 = MS_MULQ_F16(MS_LDQ_F16(line0 + 8), g01);
      MS_FLOAT16X8 acc2 = MS_MULQ_F16(MS_LDQ_F16(line0 + 16), g02);
      MS_FLOAT16X8 acc3 = MS_MULQ_F16(MS_LDQ_F16(line0 + 24), g03);
      line0 += 32;
      acc0 = MS_FMAQ_F16(acc0, MS_LDQ_F16(line1), g10);
      acc1 = MS_FMAQ_F16(acc1, MS_LDQ_F16(line1 + 8), g11);
      acc2 = MS_FMAQ_F16(acc2, MS_LDQ_F16(line1 + 16), g12);
      acc3 = MS_FMAQ_F16(acc3, MS_LDQ_F16(line1 + 24), g13);

      line1 += 32;
      acc0 = MS_FMAQ_F16(acc0, MS_LDQ_F16(line2), g20);
      acc1 = MS_FMAQ_F16(acc1, MS_LDQ_F16(line2 + 8), g21);
      acc2 = MS_FMAQ_F16(acc2, MS_LDQ_F16(line2 + 16), g22);
      acc3 = MS_FMAQ_F16(acc3, MS_LDQ_F16(line2 + 24), g23);

      line2 += 32;
      MS_FLOAT16X8 res0 = MS_ADDQ_F16(acc0, MS_ADDQ_F16(acc2, acc1));
      MS_FLOAT16X8 res1 = MS_ADDQ_F16(acc1, MS_SUBQ_F16(acc3, acc2));
      res0 = MS_ADDQ_F16(res0, bias);
      res1 = MS_ADDQ_F16(res1, bias);
      if (relu || relu6) {
        res0 = MS_MAXQ_F16(res0, MS_MOVQ_F16((float16_t)0.0));
        res1 = MS_MAXQ_F16(res1, MS_MOVQ_F16((float16_t)0.0));
      }
      if (relu6) {
        res0 = MS_MINQ_F16(res0, MS_MOVQ_F16((float16_t)6.0));
        res1 = MS_MINQ_F16(res1, MS_MOVQ_F16((float16_t)6.0));
      }
      if (channel >= 8) {
        MS_STQ_F16(cur_dst, res0);
        MS_STQ_F16(cur_dst + ori_channel, res1);
      } else {
        for (int i = 0; i < channel; i++) {
          cur_dst[i] = res0[i];
          cur_dst[ori_channel + i] = res1[i];
        }
      }
      cur_dst += 2 * ori_channel;
    }
    if (ow < width) {
      MS_FLOAT16X8 acc0 = MS_MULQ_F16(MS_LDQ_F16(line0), g00);
      MS_FLOAT16X8 acc1 = MS_MULQ_F16(MS_LDQ_F16(line0 + 8), g01);
      MS_FLOAT16X8 acc2 = MS_MULQ_F16(MS_LDQ_F16(line0 + 16), g02);
      line0 += 32;
      acc0 = MS_FMAQ_F16(acc0, MS_LDQ_F16(line1), g10);
      acc1 = MS_FMAQ_F16(acc1, MS_LDQ_F16(line1 + 8), g11);
      acc2 = MS_FMAQ_F16(acc2, MS_LDQ_F16(line1 + 16), g12);

      line1 += 32;
      acc0 = MS_FMAQ_F16(acc0, MS_LDQ_F16(line2), g20);
      acc1 = MS_FMAQ_F16(acc1, MS_LDQ_F16(line2 + 8), g21);
      acc2 = MS_FMAQ_F16(acc2, MS_LDQ_F16(line2 + 16), g22);

      line2 += 32;
      MS_FLOAT16X8 res0 = MS_ADDQ_F16(acc0, MS_ADDQ_F16(acc2, acc1));
      res0 = MS_ADDQ_F16(res0, bias);
      if (relu || relu6) {
        res0 = MS_MAXQ_F16(res0, MS_MOVQ_F16((float16_t)0.0));
      }
      if (relu6) {
        res0 = MS_MINQ_F16(res0, MS_MOVQ_F16((float16_t)6.0));
      }
      if (channel >= 8) {
        MS_STQ_F16(cur_dst, res0);
      } else {
        for (int i = 0; i < channel; i++) {
          cur_dst[i] = res0[i];
        }
      }
    }
    dst += 8;
  }
}

void ConvDw3x3Fp16(float16_t *output_data, float16_t *buffer, const float16_t *input_data, const float16_t *weight_data,
                   const float16_t *bias_data, const ConvParameter *conv_param, int start_oh, int end_oh) {
  int units = UP_DIV(conv_param->output_w_, C2NUM);
  int c8 = UP_ROUND(conv_param->input_channel_, C8NUM);
  int line = conv_param->input_channel_ * conv_param->input_w_;

  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;

  for (int b = 0; b < conv_param->output_batch_; b++) {
    const float16_t *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    float16_t *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    float16_t *line0 = buffer;
    float16_t *line1 = buffer + units * c8 * C4NUM;
    float16_t *line2 = buffer + units * c8 * C4NUM * 2;
    float16_t *lines[3] = {line0, line1, line2};
    int oh = start_oh;
    if (oh == 0) {
      // input trans
      ConvDw3x3InitTopFp16(src, lines, conv_param->output_w_, conv_param->input_channel_);
    } else {
      // input trans
      ConvDw3x3InitRowFp16(src + oh * line, lines, conv_param->output_w_, conv_param->input_channel_);
    }
    // dst calc and trans
    ConvDw3x3LineFp16(dst + oh * line, lines, weight_data, bias_data, conv_param->output_w_, conv_param->input_channel_,
                      relu, relu6);
    for (oh = start_oh + 1; oh < end_oh - 1; oh++) {
      // input trans
      ConvDw3x3RowFp16(src + oh * line + line, lines, conv_param->output_w_, conv_param->input_channel_);
      // dst calc and trans
      ConvDw3x3LineFp16(dst + oh * line, lines, weight_data, bias_data, conv_param->output_w_,
                        conv_param->input_channel_, relu, relu6);
    }
    if (oh == conv_param->output_h_ - 1) {
      // input trans
      ConvDw3x3BottomFp16(lines, conv_param->output_w_, conv_param->input_channel_);
    } else {
      // input trans
      ConvDw3x3RowFp16(src + oh * line + line, lines, conv_param->output_w_, conv_param->input_channel_);
    }
    // dst calc and trans
    ConvDw3x3LineFp16(dst + oh * line, lines, weight_data, bias_data, conv_param->output_w_, conv_param->input_channel_,
                      relu, relu6);
  }
}

#endif

void ConvDwFp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                const float16_t *bias_data, const ConvParameter *conv_param, int task_id) {
  NNACL_CHECK_ZERO_RETURN(conv_param->stride_w_);
  int h_step = UP_DIV(conv_param->output_h_, conv_param->thread_num_);
  int h_start = h_step * task_id;
  int h_end = MSMIN(h_start + h_step, conv_param->output_h_);
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    const float16_t *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    float16_t *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    for (int oh = h_start; oh < h_end; oh++) {
      float16_t *dst_data = dst + oh * conv_param->output_w_ * conv_param->output_channel_;

      int ih_origin = oh * conv_param->stride_h_ - conv_param->pad_u_;
      int start_kh = MSMAX(0, UP_DIV(-ih_origin, conv_param->dilation_h_));
      int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih_origin, conv_param->dilation_h_));

      for (int ow = 0; ow < conv_param->output_w_; ow++) {
        memcpy(dst_data + ow * conv_param->output_channel_, bias_data, conv_param->output_channel_ * sizeof(float16_t));
      }
      for (int kh = start_kh; kh < end_kh; kh++) {
        int ih = ih_origin + conv_param->dilation_h_ * kh;

        const float16_t *src_kh = src + ih * conv_param->input_w_ * conv_param->input_channel_;
        const float16_t *weight_kh = weight_data + kh * conv_param->kernel_w_ * conv_param->output_channel_;

        int in_sw_step = conv_param->stride_w_ * conv_param->input_channel_;
        for (int kw = 0; kw < conv_param->kernel_w_; kw++) {
          int out_w_start = MSMAX(
            0, (conv_param->pad_l_ - conv_param->dilation_w_ * kw + conv_param->stride_w_ - 1) / conv_param->stride_w_);
          int out_w_end = MSMIN(conv_param->output_w_, (conv_param->input_w_ + conv_param->pad_l_ -
                                                        conv_param->dilation_w_ * kw + conv_param->stride_w_ - 1) /
                                                         conv_param->stride_w_);

          float16_t *dst_w = dst_data + out_w_start * conv_param->output_channel_;
          int iw_origin = (out_w_start * conv_param->stride_w_) - conv_param->pad_l_ + conv_param->dilation_w_ * kw;

          const float16_t *src_kw = src_kh + iw_origin * conv_param->input_channel_;
          int num_pixels = out_w_end - out_w_start;
          ConvDwFp16Row(dst_w, src_kw, weight_kh, num_pixels, conv_param->output_channel_, in_sw_step);
          weight_kh += conv_param->output_channel_;
        }
      }
      if (relu) {
        ReluFp16(dst_data, dst_data, conv_param->output_w_ * conv_param->output_channel_);
      }
      if (relu6) {
        Relu6Fp16(dst_data, dst_data, conv_param->output_w_ * conv_param->output_channel_);
      }
    }
  }
}

/*conv depthwise fp16 begin*/
void DepthwiseBorderPixelFp16(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias,
                              int height, int width, int in_kh_step, int in_kw_step, int kernel_w_step, bool is_relu,
                              bool is_relu6) {
  for (int c = 0; c < C8NUM; c++) {
    dst[c] = 0;
  }
  const float16_t *src_kh = src;
  const float16_t *weight_kh = weight;
  for (int kh = 0; kh < height; kh++) {
    const float16_t *src_kw = src_kh;
    const float16_t *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
      float16x8_t src_8 = vld1q_f16(src_kw);
      float16x8_t weight_8 = vld1q_f16(weight_kw);
      float16x8_t dst_8 = vld1q_f16(dst);
      dst_8 = vfmaq_f16(dst_8, src_8, weight_8);
      vst1q_f16(dst, dst_8);

      src_kw += in_kw_step;
      weight_kw += C8NUM;
    }  // kernel_w loop
    src_kh += in_kh_step;
    weight_kh += kernel_w_step;
  }  // kernel_h loop
  for (int c = 0; c < C8NUM; c++) {
    dst[c] += bias[c];
    dst[c] = (is_relu) ? (MSMAX(0, dst[c])) : (dst[c]);
    dst[c] = (is_relu6) ? (MSMIN(6, MSMAX(0, dst[c]))) : (dst[c]);
  }
}

void DepthwiseBorderFp16(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias, int top,
                         int bottom, int left, int right, const ConvParameter *conv_param,
                         const SlidingWindowParam *sliding) {
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  float16_t *dst_h = dst + top * sliding->out_h_step_;
  for (int oh = top; oh < bottom; oh++) {
    int ih = oh * conv_param->stride_h_ - conv_param->pad_u_;
    int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));
    const float16_t *src_h = src + ih * sliding->in_h_step_;

    float16_t *dst_kernel = dst_h + left * sliding->block_channel_;
    for (int ow = left; ow < right; ow++) {
      int iw = ow * conv_param->stride_w_ - conv_param->pad_l_;
      int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));
      const float16_t *src_w = src_h + iw * sliding->block_channel_;

      const float16_t *src_kernel = src_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
      const float16_t *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C8NUM;
#ifdef ENABLE_ARM64
      ConvDwFp16Border(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                       sliding->in_kh_step_ * sizeof(float16_t), sliding->in_kw_step_ * sizeof(float16_t),
                       conv_param->kernel_w_ * C8NUM * sizeof(float16_t), relu, relu6);
#else
      DepthwiseBorderPixelFp16(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                               sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_ * C8NUM, relu, relu6);
#endif
      dst_kernel += sliding->block_channel_;
    }  // width loop
    dst_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void DepthwiseCenterFp16(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias,
                         int height, int width, int kernel_h, int kernel_w, int out_h_step, int block_channel,
                         int in_sh_step, int in_sw_step, int in_kh_step, int in_kw_step, bool is_relu, bool is_relu6) {
  float16_t *dst_h = dst;
  const float16_t *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float16_t *dst_w = dst_h;
    const float16_t *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      const float16_t *src_kh = src_w;
      const float16_t *weight_kh = weight;
      for (int c = 0; c < C8NUM; c++) {
        dst_w[c] = 0;
      }
      for (int kh = 0; kh < kernel_h; kh++) {
        const float16_t *src_kw = src_kh;
        const float16_t *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++) {
#ifdef ENABLE_ARM64
          float16x8_t src_8 = vld1q_f16(src_kw);
          float16x8_t weight_8 = vld1q_f16(weight_kw);
          float16x8_t dst_8 = vld1q_f16(dst_w);
          dst_8 = vfmaq_f16(dst_8, src_8, weight_8);
          vst1q_f16(dst_w, dst_8);
#else
          for (int c = 0; c < C8NUM; c++) {
            dst_w[c] += src_kw[c] * weight_kw[c];
          }
#endif
          src_kw += in_kw_step;
          weight_kw += C8NUM;
        }  // kernel_w loop
        src_kh += in_kh_step;
        weight_kh += kernel_w * C8NUM;
      }  // kernel_h loop
      // add biad relu
      for (int c = 0; c < C8NUM; c++) {
        dst_w[c] += bias[c];
        dst_w[c] = (is_relu) ? (MSMAX(0, dst_w[c])) : (dst_w[c]);
        dst_w[c] = (is_relu6) ? (MSMIN(6, MSMAX(0, dst_w[c]))) : (dst_w[c]);
      }
      dst_w += block_channel;
      src_w += in_sw_step;
    }  // dst_width loop
    dst_h += out_h_step;
    src_h += in_sh_step;
  }  // dst_height loop
}
#endif

// conv depthwise fp16: sliding window
void ConvDwC8Fp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                  const float16_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                  int task_id) {
  NNACL_CHECK_ZERO_RETURN(conv_param->dilation_h_);
  NNACL_CHECK_ZERO_RETURN(conv_param->dilation_w_);
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  const float16_t *src = input_data;
  float16_t *dst = output_data;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      const float16_t *src_data = src + oc * C8NUM;
      float16_t *dst_data = dst + oc * C8NUM;
      const float16_t *weight = weight_data + oc * sliding->kernel_step_;
      const float16_t *bias = bias_data + oc * C8NUM;
      DepthwiseBorderFp16(dst_data, src_data, weight, bias, 0, sliding->top_, 0, conv_param->output_w_, conv_param,
                          sliding);
      DepthwiseBorderFp16(dst_data, src_data, weight, bias, sliding->bottom_, conv_param->output_h_, 0,
                          conv_param->output_w_, conv_param, sliding);
      DepthwiseBorderFp16(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, 0, sliding->left_,
                          conv_param, sliding);
      DepthwiseBorderFp16(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, sliding->right_,
                          conv_param->output_w_, conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int in_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int in_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        const float16_t *in_t = src_data + in_h_start * sliding->in_h_step_ + in_w_start * sliding->block_channel_;
        float16_t *out_t = dst_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
#ifdef ENABLE_ARM64
        ConvDwFp16Center(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                         conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(float16_t),
                         sliding->block_channel_ * sizeof(float16_t), sliding->in_sh_step_ * sizeof(float16_t),
                         sliding->in_sw_step_ * sizeof(float16_t), sliding->in_kh_step_ * sizeof(float16_t),
                         sliding->in_kw_step_ * sizeof(float16_t), relu, relu6);
#else
        DepthwiseCenterFp16(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_,
                            sliding->right_ - sliding->left_, conv_param->kernel_h_, conv_param->kernel_w_,
                            sliding->out_h_step_, sliding->block_channel_, sliding->in_sh_step_, sliding->in_sw_step_,
                            sliding->in_kh_step_, sliding->in_kw_step_, relu, relu6);
#endif
      }
    }  // output C8 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nchwc8
}
/*conv depthwise fp16 end*/

/*deconv depthwise fp16 begin*/
void DeconvDepthwiseBorderPixelFp16(float16_t *dst, const float16_t *src, const float16_t *weight, int height,
                                    int width, int in_kh_step, int in_kw_step, int kernel_w_step) {
  float16_t *dst_kh = dst;
  const float16_t *weight_kh = weight;
  for (int kh = 0; kh < height; kh++) {
    float16_t *dst_kw = dst_kh;
    const float16_t *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
      float16x8_t src_8 = vld1q_f16(src);
      float16x8_t weight_8 = vld1q_f16(weight_kw);
      float16x8_t dst_8 = vld1q_f16(dst_kw);
      dst_8 = vfmaq_f16(dst_8, src_8, weight_8);
      vst1q_f16(dst_kw, dst_8);

      dst_kw += in_kw_step;
      weight_kw += C8NUM;
    }  // kernel_w loop
    dst_kh += in_kh_step;
    weight_kh += kernel_w_step;
  }  // kernel_h loop
}

void DeconvDepthwiseBorderFp16(float16_t *dst, const float16_t *src, const float16_t *weight, int top, int bottom,
                               int left, int right, const ConvParameter *conv_param,
                               const SlidingWindowParam *sliding) {
  const float16_t *src_h = src + top * sliding->out_h_step_;
  for (int ih = top; ih < bottom; ih++) {
    int oh = ih * conv_param->stride_h_ - conv_param->pad_u_;
    int start_kh = MSMAX(0, UP_DIV(-oh, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->output_h_ - oh, conv_param->dilation_h_));
    float16_t *dst_h = dst + oh * sliding->in_h_step_;

    const float16_t *src_kernel = src_h + left * sliding->block_channel_;
    for (int iw = left; iw < right; iw++) {
      int ow = iw * conv_param->stride_w_ - conv_param->pad_l_;
      int start_kw = MSMAX(0, UP_DIV(-ow, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->output_w_ - ow, conv_param->dilation_w_));
      float16_t *dst_w = dst_h + ow * sliding->block_channel_;

      const float16_t *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C8NUM;
      float16_t *dst_kernel = dst_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
#ifdef ENABLE_ARM64
      DeconvDwFp16Border(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                         sliding->in_kh_step_ * sizeof(float16_t), sliding->in_kw_step_ * sizeof(float16_t),
                         conv_param->kernel_w_ * C8NUM * sizeof(float16_t));
#else
      DeconvDepthwiseBorderPixelFp16(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                                     sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_ * C8NUM);
#endif
      src_kernel += sliding->block_channel_;
    }  // width loop
    src_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void DeconvDepthwiseCenterFp16(float16_t *dst, const float16_t *src, const float16_t *weight, int height, int width,
                               int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step,
                               int in_sw_step, int in_kh_step, int in_kw_step) {
  float16_t *dst_h = dst;
  const float16_t *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float16_t *dst_w = dst_h;
    const float16_t *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      float16_t *dst_kh = dst_w;
      const float16_t *weight_kh = weight;
      for (int kh = 0; kh < kernel_h; kh++) {
        float16_t *dst_kw = dst_kh;
        const float16_t *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++) {
#ifdef ENABLE_NEON
          float16x8_t src_8 = vld1q_f16(src_w);
          float16x8_t weight_8 = vld1q_f16(weight_kw);
          float16x8_t dst_8 = vld1q_f16(dst_kw);
          dst_8 = vfmaq_f16(dst_8, src_8, weight_8);
          vst1q_f16(dst_kw, dst_8);
#else
          for (int c = 0; c < C8NUM; c++) {
            dst_kw[c] += src_w[c] * weight_kw[c];
          }
#endif
          dst_kw += in_kw_step;
          weight_kw += C8NUM;
        }  // kernel_w loop
        dst_kh += in_kh_step;
        weight_kh += kernel_w * C8NUM;
      }  // kernel_h loop
      dst_w += in_sw_step;
      src_w += block_channel;
    }  // dst_width loop
    dst_h += in_sh_step;
    src_h += out_h_step;
  }  // dst_height loop
}
#endif

void DeconvDepthwisePostFuncFp16(float16_t *dst, const float16_t *bias, int block_channel,
                                 const ConvParameter *conv_param) {
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  float16_t *dst_k = dst;
  for (int k = 0; k < conv_param->output_h_ * conv_param->output_w_; k++) {
    for (int c = 0; c < C8NUM; c++) {
      dst_k[c] += bias[c];
      dst_k[c] = (relu) ? (MSMAX(0, dst_k[c])) : (dst_k[c]);
      dst_k[c] = (relu6) ? (MSMIN(6, MSMAX(0, dst_k[c]))) : (dst_k[c]);
    }
    dst_k += block_channel;
  }
}

// deconv depthwise fp16: sliding window
void DeconvDwC8Fp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                    const float16_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                    int task_id) {
  const float16_t *src = input_data;
  float16_t *dst = output_data;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      const float16_t *src_data = src + oc * C8NUM;
      float16_t *dst_data = dst + oc * C8NUM;
      const float16_t *weight = weight_data + oc * sliding->kernel_step_;
      const float16_t *bias = bias_data + oc * C8NUM;
      DeconvDepthwiseBorderFp16(dst_data, src_data, weight, 0, sliding->top_, 0, conv_param->input_w_, conv_param,
                                sliding);
      DeconvDepthwiseBorderFp16(dst_data, src_data, weight, sliding->bottom_, conv_param->input_h_, 0,
                                conv_param->input_w_, conv_param, sliding);
      DeconvDepthwiseBorderFp16(dst_data, src_data, weight, sliding->top_, sliding->bottom_, 0, sliding->left_,
                                conv_param, sliding);
      DeconvDepthwiseBorderFp16(dst_data, src_data, weight, sliding->top_, sliding->bottom_, sliding->right_,
                                conv_param->input_w_, conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int oh_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int oh_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        float16_t *out_t = dst_data + oh_h_start * sliding->in_h_step_ + oh_w_start * sliding->block_channel_;
        const float16_t *in_t =
          src_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
#ifdef ENABLE_ARM64
        DeconvDwFp16Center(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                           conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(float16_t),
                           sliding->block_channel_ * sizeof(float16_t), sliding->in_sh_step_ * sizeof(float16_t),
                           sliding->in_sw_step_ * sizeof(float16_t), sliding->in_kh_step_ * sizeof(float16_t),
                           sliding->in_kw_step_ * sizeof(float16_t));
#else
        DeconvDepthwiseCenterFp16(out_t, in_t, weight, sliding->bottom_ - sliding->top_,
                                  sliding->right_ - sliding->left_, conv_param->kernel_h_, conv_param->kernel_w_,
                                  sliding->out_h_step_, sliding->block_channel_, sliding->in_sh_step_,
                                  sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_);
#endif
      }
      DeconvDepthwisePostFuncFp16(dst_data, bias, sliding->block_channel_, conv_param);
    }  // output C8 loop
    src += sliding->out_step_;
    dst += sliding->in_step_;
  }  // batch loop
  // output nchwc8
}
/*deconv depthwise fp16 end*/
