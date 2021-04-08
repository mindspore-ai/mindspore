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

#include "nnacl/int8/conv_depthwise_int8.h"
#include <string.h>
#include "nnacl/int8/fixed_point.h"
#include "nnacl/int8/common_func_int8.h"

/*conv depthwise int8 begin*/
#ifndef ENABLE_ARM
void ConvDwInt8Row(int32_t *output_ptr, const int8_t *input_ptr, const int16_t *weight_ptr, int num_pixels,
                   int output_channel, int input_step, int8_t input_zp) {
  for (int i = 0; i < num_pixels; i++) {
    for (int c = 0; c < output_channel; c++) {
      const int16_t input = input_ptr[c] - input_zp;
      *output_ptr++ += input * weight_ptr[c];
    }
    input_ptr += input_step;
  }
}
#endif

void ConvDwInt8Post(int8_t *dst, int32_t *buffer, int output_w, int channel, int32_t output_zp, int32_t *out_multiplier,
                    int32_t *left_shift, int32_t *right_shift, int32_t acc_min, int32_t acc_max, bool per_channel) {
  if (per_channel) {
    // support perchannel
    for (int w = 0; w < output_w; w++) {
      int channel4 = 0;
#ifdef ENABLE_ARM
      channel4 = channel / 4 * 4;
      ConvDwInt8PostAlign4PerChannel(dst, buffer, channel4, output_zp, out_multiplier, left_shift, right_shift, acc_min,
                                     acc_max);
#endif
      for (int c = channel4; c < channel; c++) {
        buffer[c] = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(buffer[c] * (1 << (unsigned int)left_shift[c]), out_multiplier[c]),
          -right_shift[c]);
        buffer[c] += output_zp;
        buffer[c] = MSMAX(buffer[c], acc_min);
        buffer[c] = MSMIN(buffer[c], acc_max);
        dst[c] = (buffer[c]);
      }
      buffer += channel;
      dst += channel;
    }
  } else {
    int num_pixels = output_w * channel;
    int align_num = 0;
#ifdef ENABLE_ARM
    align_num = num_pixels / 4 * 4;
    ConvDwInt8PostAlign4(dst, buffer, align_num, output_zp, out_multiplier[0], left_shift[0], right_shift[0], acc_min,
                         acc_max);
#endif
    for (int i = align_num; i < num_pixels; i++) {
      buffer[i] = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(buffer[i] * (1 << (unsigned int)left_shift[0]), out_multiplier[0]),
        -right_shift[0]);
      buffer[i] += output_zp;
      buffer[i] = MSMAX(buffer[i], acc_min);
      buffer[i] = MSMIN(buffer[i], acc_max);
      dst[i] = (buffer[i]);
    }
  }
}

void ConvDwInt8(int8_t *output_data, int32_t *row_buffer, const int8_t *input_data, const int16_t *weight_data,
                const int32_t *bias_data, const ConvParameter *conv_param, int task_id) {
  int step_h = UP_DIV(conv_param->output_h_, conv_param->thread_num_);
  int start_h = step_h * task_id;
  int end_h = MSMIN(start_h + step_h, conv_param->output_h_);

  bool filter_per_channel = conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
  int *out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_;
  int *left_shift = conv_param->conv_quant_arg_.left_shift_;
  int *right_shift = conv_param->conv_quant_arg_.right_shift_;

  int intput_zp = conv_param->conv_quant_arg_.input_quant_args_[0].zp_;
  int output_zp = conv_param->conv_quant_arg_.output_quant_args_[0].zp_;
  int acc_min = conv_param->conv_quant_arg_.out_act_min_[0];
  int acc_max = conv_param->conv_quant_arg_.out_act_max_[0];

  for (int b = 0; b < conv_param->output_batch_; b++) {
    const int8_t *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    int8_t *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    for (int oh = start_h; oh < end_h; oh++) {
      int8_t *dst_data = dst + oh * conv_param->output_w_ * conv_param->output_channel_;

      int ih_origin = oh * conv_param->stride_h_ - conv_param->pad_u_;
      int start_kh = MSMAX(0, UP_DIV(-ih_origin, conv_param->dilation_h_));
      int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih_origin, conv_param->dilation_h_));

      // init acc
      for (int ow = 0; ow < conv_param->output_w_; ow++) {
        memcpy(row_buffer + ow * conv_param->output_channel_, bias_data, conv_param->output_channel_ * sizeof(int32_t));
      }
      for (int kh = start_kh; kh < end_kh; kh++) {
        int ih = ih_origin + conv_param->dilation_w_ * kh;

        const int8_t *src_kh = src + ih * conv_param->input_w_ * conv_param->input_channel_;
        const int16_t *weight_kh = weight_data + kh * conv_param->kernel_w_ * conv_param->output_channel_;

        int in_sw_step = conv_param->stride_w_ * conv_param->input_channel_;
        for (int kw = 0; kw < conv_param->kernel_w_; kw++) {
          int out_w_start = MSMAX(
            0, (conv_param->pad_l_ - conv_param->dilation_w_ * kw + conv_param->stride_w_ - 1) / conv_param->stride_w_);
          int out_w_end = MSMIN(conv_param->output_w_, (conv_param->input_w_ + conv_param->pad_l_ -
                                                        conv_param->dilation_w_ * kw + conv_param->stride_w_ - 1) /
                                                         conv_param->stride_w_);

          int32_t *acc_w = row_buffer + out_w_start * conv_param->output_channel_;
          int iw_origin = (out_w_start * conv_param->stride_w_) - conv_param->pad_l_ + conv_param->dilation_w_ * kw;

          const int8_t *src_kw = src_kh + iw_origin * conv_param->input_channel_;
          int num_pixels = out_w_end - out_w_start;

          ConvDwInt8Row(acc_w, src_kw, weight_kh, num_pixels, conv_param->output_channel_, in_sw_step, intput_zp);
          weight_kh += conv_param->output_channel_;
        }
      }
      // post func, acc int32 -> dst int8
      ConvDwInt8Post(dst_data, row_buffer, conv_param->output_w_, conv_param->output_channel_, output_zp,
                     out_multiplier, left_shift, right_shift, acc_min, acc_max, filter_per_channel);
    }
  }
}
/*conv depthwise int8 end*/

/*conv depthwise 3x3 int8 begin*/
void ConvDw3x3Int8InitBuffer(int8_t *buffer, const int8_t *input, const ConvParameter *conv_param, int block_input_h,
                             int block_input_w) {
  for (int h = 0; h < block_input_h; h++) {
    const int8_t *src = input;
    for (int w = 0; w < block_input_w; w++) {
      memcpy(buffer, src, 64);
      src += conv_param->input_channel_;
      buffer += 64;
    }
    input += conv_param->input_w_ * conv_param->input_channel_;
  }
}

void ConvDw3x3Int8Window(int8_t *output, const int8_t *buffer, const int16_t *weight, const int32_t *bias, int col_size,
                         int row_size, int channel, int output_h, int output_w, int8_t in_zp, int32_t out_zp,
                         int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift, int32_t acc_min,
                         int32_t acc_max, int stride, bool per_channel) {
  for (int w = 0; w < output_w; w++) {
    int tmp_buffer[C8NUM];
    for (int i = 0; i < C8NUM; i++) {
      tmp_buffer[i] = 0;
    }
    int8_t *output_tmp = output;
    const int8_t *src_kh = buffer;
    const int16_t *weight_kh = weight;
    for (int kh = 0; kh < 3; kh++) {
      const int8_t *src_kw = src_kh;
      const int16_t *weight_kw = weight_kh;
      for (int kw = 0; kw < 3; kw++) {
        for (int c = 0; c < 8; c++) {
          tmp_buffer[c] += (src_kw[c] - in_zp) * weight_kw[c];
        }
        src_kw += col_size;
        weight_kw += channel;
      }
      src_kh += row_size;
      weight_kh += 3 * channel;
    }
    if (per_channel) {
      for (int c = 0; c < C8NUM; c++) {
        tmp_buffer[c] += bias[c];
        tmp_buffer[c] = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(tmp_buffer[c] * (1 << (unsigned int)left_shift[c]), out_multiplier[c]),
          -right_shift[c]);
        tmp_buffer[c] += out_zp;
        tmp_buffer[c] = MSMAX(tmp_buffer[c], acc_min);
        tmp_buffer[c] = MSMIN(tmp_buffer[c], acc_max);
        *output_tmp++ = (tmp_buffer[c]);
      }
    } else {
      for (int c = 0; c < C8NUM; c++) {
        tmp_buffer[c] += bias[c];
        tmp_buffer[c] = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(tmp_buffer[c] * (1 << (unsigned int)left_shift[0]), out_multiplier[0]),
          -right_shift[0]);
        tmp_buffer[c] += out_zp;
        tmp_buffer[c] = MSMAX(tmp_buffer[c], acc_min);
        tmp_buffer[c] = MSMIN(tmp_buffer[c], acc_max);
        *output_tmp++ = (tmp_buffer[c]);
      }
    }
    output += channel;
    buffer += col_size * stride;
  }
}

void ConvDw3x3Int8Block(int8_t *output, const int8_t *buffer, const int16_t *weight, const int32_t *bias, int start_c,
                        int end_c, int col_size, int row_size, int channel, int output_h, int output_w, int8_t in_zp,
                        int32_t out_zp, int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift,
                        int32_t acc_min, int32_t acc_max, int stride, bool per_channel) {
  for (; start_c <= end_c - 8; start_c += 8) {
#ifdef ENABLE_ARM64
    if (stride == 1) {
      ConvDw3x3Int8Neon64(output, buffer, weight, bias, col_size, row_size, channel, output_h, output_w, in_zp, out_zp,
                          out_multiplier, left_shift, right_shift, acc_min, acc_max, per_channel);
    } else {
      ConvDw3x3Int8Stride2(output, buffer, weight, bias, col_size, row_size, channel, output_h, output_w, in_zp, out_zp,
                           out_multiplier, left_shift, right_shift, acc_min, acc_max, per_channel);
    }

#else
    ConvDw3x3Int8Window(output, buffer, weight, bias, col_size, row_size, channel, output_h, output_w, in_zp, out_zp,
                        out_multiplier, left_shift, right_shift, acc_min, acc_max, stride, per_channel);
#endif
    output += 8;
    buffer += 8;
    weight += 8;
    bias += 8;
    if (per_channel) {
      out_multiplier += 8;
      left_shift += 8;
      right_shift += 8;
    }
  }
}

void ConvDw3x3Int8Row(int8_t *output, int8_t *buffer, const int8_t *input, const int16_t *weight, const int32_t *bias,
                      const ConvParameter *conv_param, int start_w, int end_w, int block_output_h, int block_output_w,
                      int block_input_h, int block_input_w) {
  bool filter_per_channel = conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
  int *out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_;
  int *left_shift = conv_param->conv_quant_arg_.left_shift_;
  int *right_shift = conv_param->conv_quant_arg_.right_shift_;
  int in_zp = conv_param->conv_quant_arg_.input_quant_args_[0].zp_;
  int out_zp = conv_param->conv_quant_arg_.output_quant_args_[0].zp_;
  int acc_min = conv_param->conv_quant_arg_.out_act_min_[0];
  int acc_max = conv_param->conv_quant_arg_.out_act_max_[0];

  const int ih_offset = 64 * block_input_w;
  int w = start_w;
  if (conv_param->output_channel_ > 64 || (conv_param->output_channel_ < 64 && conv_param->input_w_ > 150)) {
    for (; w <= end_w - block_output_w; w += block_output_w) {
      int8_t *output_ptr = output;
      const int8_t *input_ptr = input;
      const int16_t *weight_ptr = weight;
      const int32_t *bias_ptr = bias;
      int32_t *out_multiplier_ptr = out_multiplier;
      int32_t *left_shift_ptr = left_shift;
      int32_t *right_shift_ptr = right_shift;
      int c = 0;
      for (; c <= conv_param->output_channel_ - 64; c += 64) {
        ConvDw3x3Int8InitBuffer(buffer, input_ptr, conv_param, block_input_h, block_input_w);
        ConvDw3x3Int8Block(output_ptr, buffer, weight_ptr, bias_ptr, 0, 64, 64, ih_offset, conv_param->input_channel_,
                           block_output_h, block_output_w, in_zp, out_zp, out_multiplier_ptr, left_shift_ptr,
                           right_shift_ptr, acc_min, acc_max, conv_param->stride_h_, filter_per_channel);
        output_ptr += 64;
        input_ptr += 64;
        weight_ptr += 64;
        bias_ptr += 64;
        if (filter_per_channel) {
          out_multiplier_ptr += 64;
          left_shift_ptr += 64;
          right_shift_ptr += 64;
        }
      }
      // left channel
      ConvDw3x3Int8Block(output_ptr, input_ptr, weight_ptr, bias_ptr, c, conv_param->input_channel_,
                         conv_param->input_channel_, conv_param->input_w_ * conv_param->input_channel_,
                         conv_param->input_channel_, block_output_h, block_output_w, in_zp, out_zp, out_multiplier_ptr,
                         left_shift_ptr, right_shift_ptr, acc_min, acc_max, conv_param->stride_h_, filter_per_channel);
      output += block_output_w * conv_param->input_channel_;
      input += conv_param->stride_w_ * block_output_w * conv_param->input_channel_;
    }
  }
  // left width
  int left_width = end_w - w;
  if (left_width > 0) {
    ConvDw3x3Int8Block(output, input, weight, bias, 0, conv_param->input_channel_, conv_param->input_channel_,
                       conv_param->input_w_ * conv_param->input_channel_, conv_param->input_channel_, block_output_h,
                       left_width, in_zp, out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max,
                       conv_param->stride_h_, filter_per_channel);
  }
}

void ConvDw3x3Int8(int8_t *output_data, int8_t *buffer, const int8_t *input_data, const int16_t *weight_data,
                   const int32_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                   int task_id) {
  int output_h = sliding->bottom_ - sliding->top_;
  int step_oh = UP_DIV(output_h, conv_param->thread_num_);
  int start_oh = step_oh * task_id + sliding->top_;
  int end_oh = MSMIN(start_oh + step_oh, sliding->bottom_);
  int start_ow = sliding->left_;
  int end_ow = sliding->right_;

  const int block_output_h = 1;
  int block_output_w = conv_param->stride_w_ == 1 ? 30 : 14;
  const int block_input_h = 3;
  int block_input_w = conv_param->stride_w_ * (block_output_w - 1) + 3;

  for (int b = 0; b < conv_param->output_batch_; b++) {
    int start_ih = start_oh * conv_param->stride_h_ - conv_param->pad_u_;
    int start_iw = start_ow * conv_param->stride_w_ - conv_param->pad_l_;
    const int8_t *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_ +
                        start_ih * conv_param->input_w_ * conv_param->input_channel_ +
                        start_iw * conv_param->input_channel_;
    int8_t *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_ +
                  start_oh * conv_param->output_w_ * conv_param->output_channel_ +
                  start_ow * conv_param->output_channel_;

    for (int oh = start_oh; oh < end_oh; oh++) {
      ConvDw3x3Int8Row(dst, buffer, src, weight_data, bias_data, conv_param, start_ow, end_ow, block_output_h,
                       block_output_w, block_input_h, block_input_w);
      src += conv_param->stride_h_ * conv_param->input_w_ * conv_param->input_channel_;
      dst += conv_param->output_w_ * conv_param->output_channel_;
    }
  }
}

#ifndef ENABLE_ARM32
void ConvDw3x3Int8BorderPixel(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int height,
                              int width, int in_kh_step, int in_kw_step, int channel, int8_t in_zp, int32_t out_zp,
                              int *out_multiplier, int *left_shift, int *right_shift, int32_t acc_min, int32_t acc_max,
                              bool per_channel) {
  for (int c = 0; c < channel; c += 8) {
    int tmp_buffer[8];
    for (int i = 0; i < 8; i++) {
      tmp_buffer[i] = 0;
    }
    const int8_t *src_kh = src;
    const int16_t *weight_kh = weight;
    for (int kh = 0; kh < height; kh++) {
      const int8_t *src_kw = src_kh;
      const int16_t *weight_kw = weight_kh;
      for (int kw = 0; kw < width; kw++) {
        for (int i = 0; i < 8; i++) {
          tmp_buffer[i] += (src_kw[c + i] - in_zp) * weight_kw[c + i];
        }
        src_kw += in_kw_step;
        weight_kw += channel;
      }  // kernel_w loop
      src_kh += in_kh_step;
      weight_kh += 3 * channel;
    }  // kernel_h loop
    if (per_channel) {
      for (int i = 0; i < 8; i++) {
        tmp_buffer[i] += bias[c + i];
        tmp_buffer[i] = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(tmp_buffer[i] * (1 << (unsigned int)left_shift[i]), out_multiplier[i]),
          -right_shift[i]);
        tmp_buffer[i] += out_zp;
        tmp_buffer[i] = MSMAX(tmp_buffer[i], acc_min);
        tmp_buffer[i] = MSMIN(tmp_buffer[i], acc_max);
        dst[i] = (tmp_buffer[i]);
      }
      left_shift += 8;
      right_shift += 8;
      out_multiplier += 8;
    } else {
      for (int i = 0; i < 8; i++) {
        tmp_buffer[i] += bias[c + i];
        tmp_buffer[i] = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(tmp_buffer[i] * (1 << (unsigned int)left_shift[0]), out_multiplier[0]),
          -right_shift[0]);
        tmp_buffer[i] += out_zp;
        tmp_buffer[i] = MSMAX(tmp_buffer[i], acc_min);
        tmp_buffer[i] = MSMIN(tmp_buffer[i], acc_max);
        dst[i] = (tmp_buffer[i]);
      }
    }
    dst += 8;
  }
}
#endif

#ifndef ENABLE_ARM64
void ConvDw3x3Int8Corner(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int in_kh_step,
                         int in_kw_step, int channel, int8_t in_zp, int32_t out_zp, int *out_multiplier,
                         int *left_shift, int *right_shift, int32_t acc_min, int32_t acc_max, bool per_channel) {
  ConvDw3x3Int8BorderPixel(dst, src, weight, bias, 2, 2, in_kh_step, in_kw_step, channel, in_zp, out_zp, out_multiplier,
                           left_shift, right_shift, acc_min, acc_max, per_channel);
}

void ConvDw3x3Int8Vertical(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int in_kh_step,
                           int in_kw_step, int channel, int8_t in_zp, int32_t out_zp, int *out_multiplier,
                           int *left_shift, int *right_shift, int32_t acc_min, int32_t acc_max, bool per_channel) {
  ConvDw3x3Int8BorderPixel(dst, src, weight, bias, 2, 3, in_kh_step, in_kw_step, channel, in_zp, out_zp, out_multiplier,
                           left_shift, right_shift, acc_min, acc_max, per_channel);
}

void ConvDw3x3Int8Horizontal(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int in_kh_step,
                             int in_kw_step, int channel, int8_t in_zp, int32_t out_zp, int *out_multiplier,
                             int *left_shift, int *right_shift, int32_t acc_min, int32_t acc_max, bool per_channel) {
  ConvDw3x3Int8BorderPixel(dst, src, weight, bias, 3, 2, in_kh_step, in_kw_step, channel, in_zp, out_zp, out_multiplier,
                           left_shift, right_shift, acc_min, acc_max, per_channel);
}
#endif

void ConvDw3x3Int8Pad(int8_t *output_data, const int8_t *input_data, const int16_t *weight_data,
                      const int32_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  bool filter_per_channel = conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
  int *out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_;
  int *left_shift = conv_param->conv_quant_arg_.left_shift_;
  int *right_shift = conv_param->conv_quant_arg_.right_shift_;
  int in_zp = conv_param->conv_quant_arg_.input_quant_args_[0].zp_;
  int out_zp = conv_param->conv_quant_arg_.output_quant_args_[0].zp_;
  int acc_min = conv_param->conv_quant_arg_.out_act_min_[0];
  int acc_max = conv_param->conv_quant_arg_.out_act_max_[0];
  int input_row_size = conv_param->input_w_ * conv_param->input_channel_;
  int weight_row_size = conv_param->kernel_w_ * conv_param->input_channel_;
  int output_row_size = conv_param->output_w_ * conv_param->output_channel_;
  int in_kh_step = sliding->in_kh_step_;
  int in_kw_step = sliding->in_kw_step_;

  // top
  for (int b = 0; b < conv_param->output_batch_; b++) {
    const int8_t *input_batch =
      input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    int8_t *output_batch =
      output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;

    const int8_t *input = input_batch;
    const int16_t *weight = weight_data + weight_row_size + conv_param->input_channel_;
    int8_t *output = output_batch;
    ConvDw3x3Int8Corner(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_, in_zp,
                        out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max, filter_per_channel);
    input += (conv_param->stride_w_ - 1) * conv_param->input_channel_;
    weight = weight_data + weight_row_size;
    output += conv_param->output_channel_;
    for (int out_w = sliding->left_; out_w < sliding->right_; out_w++) {
      ConvDw3x3Int8Vertical(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_, in_zp,
                            out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max, filter_per_channel);
      input += conv_param->stride_w_ * conv_param->input_channel_;
      output += conv_param->output_channel_;
    }
    ConvDw3x3Int8Corner(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_, in_zp,
                        out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max, filter_per_channel);

    // left
    input = input_batch + (conv_param->stride_h_ - 1) * input_row_size;
    weight = weight_data + conv_param->input_channel_;
    output = output_batch + output_row_size;
    for (int out_h = sliding->top_; out_h < sliding->bottom_; out_h++) {
      ConvDw3x3Int8Horizontal(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_,
                              in_zp, out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max,
                              filter_per_channel);
      input += conv_param->stride_h_ * input_row_size;
      output += output_row_size;
    }

    // right
    input = input_batch + (conv_param->input_w_ - 2) * conv_param->input_channel_ +
            (conv_param->stride_h_ - 1) * input_row_size;
    weight = weight_data;
    output = output_batch + output_row_size + (conv_param->output_w_ - 1) * conv_param->output_channel_;
    for (int out_h = sliding->top_; out_h < sliding->bottom_; out_h++) {
      ConvDw3x3Int8Horizontal(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_,
                              in_zp, out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max,
                              filter_per_channel);
      input += conv_param->stride_h_ * input_row_size;
      output += output_row_size;
    }

    // bottom
    input = input_batch + (conv_param->input_h_ - 2) * input_row_size;
    weight = weight_data + conv_param->input_channel_;
    output = output_batch + (conv_param->output_h_ - 1) * output_row_size;
    ConvDw3x3Int8Corner(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_, in_zp,
                        out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max, filter_per_channel);
    input += conv_param->stride_w_ == 1 ? 0 : conv_param->input_channel_;
    weight = weight_data;
    output += conv_param->output_channel_;
    for (int out_w = sliding->left_; out_w < sliding->right_; out_w++) {
      ConvDw3x3Int8Vertical(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_, in_zp,
                            out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max, filter_per_channel);
      input += conv_param->stride_w_ * conv_param->input_channel_;
      output += conv_param->output_channel_;
    }
    ConvDw3x3Int8Corner(output, input, weight, bias_data, in_kh_step, in_kw_step, conv_param->input_channel_, in_zp,
                        out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max, filter_per_channel);
  }
}
/*conv depthwise 3x3 int8 end*/

/*conv depthwise sliding window perchannel int8 begin*/
void ConvDwInt8BorderPixel(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int height,
                           int width, int in_kh_step, int in_kw_step, int kernel_w, int8_t *input_zp, int32_t *out_zp,
                           const int *out_multiplier, const int *left_shift, const int *right_shift, int32_t *acc_min,
                           int32_t *acc_max) {
  int tmp_buffer[C8NUM];
  for (int i = 0; i < C8NUM; i++) {
    tmp_buffer[i] = 0;
  }
  const int8_t *src_kh = src;
  const int16_t *weight_kh = weight;
  for (int kh = 0; kh < height; kh++) {
    const int8_t *src_kw = src_kh;
    const int16_t *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
      for (int c = 0; c < C8NUM; c++) {
        tmp_buffer[c] += (src_kw[c] - input_zp[c]) * weight_kw[c];
      }
      src_kw += in_kw_step;
      weight_kw += C8NUM;
    }  // kernel_w loop
    src_kh += in_kh_step;
    weight_kh += kernel_w * C8NUM;
  }  // kernel_h loop

  for (int c = 0; c < C8NUM; c++) {
    tmp_buffer[c] += bias[c];
    tmp_buffer[c] = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(tmp_buffer[c] * (1 << (unsigned int)left_shift[c]), out_multiplier[c]),
      -right_shift[c]);
    tmp_buffer[c] += out_zp[c];
    tmp_buffer[c] = MSMAX(tmp_buffer[c], acc_min[c]);
    tmp_buffer[c] = MSMIN(tmp_buffer[c], acc_max[c]);
    dst[c] = (tmp_buffer[c]);
  }
}

void ConvDwInt8Border(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int top, int bottom,
                      int left, int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                      int8_t *in_zp, int32_t *out_zp, const int *out_multiplier, const int *left_shift,
                      const int *right_shift, int32_t *acc_min, int32_t *acc_max) {
  int8_t *dst_h = dst + top * sliding->out_h_step_;
  for (int oh = top; oh < bottom; oh++) {
    int ih = oh * conv_param->stride_h_ - conv_param->pad_u_;
    int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));
    const int8_t *src_h = src + ih * sliding->in_h_step_;

    int8_t *dst_kernel = dst_h + left * sliding->block_channel_;
    for (int ow = left; ow < right; ow++) {
      int iw = ow * conv_param->stride_w_ - conv_param->pad_l_;
      int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));
      const int8_t *src_w = src_h + iw * sliding->block_channel_;

      const int8_t *src_kernel = src_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
      const int16_t *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C8NUM;

      ConvDwInt8BorderPixel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                            sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_, in_zp, out_zp,
                            out_multiplier, left_shift, right_shift, acc_min, acc_max);

      dst_kernel += sliding->block_channel_;
    }  // width loop
    dst_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM
void ConvDwInt8Center(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int height, int width,
                      int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step,
                      int in_kh_step, int in_kw_step, int8_t *in_zp, int32_t *out_zp, int32_t *out_multiplier,
                      int32_t *left_shift, int32_t *right_shift, int32_t *acc_min, int32_t *acc_max) {
  int tmp_buffer[C8NUM];
  int8_t *dst_h = dst;
  const int8_t *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    int8_t *dst_w = dst_h;
    const int8_t *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      const int8_t *src_kh = src_w;
      const int16_t *weight_kh = weight;

      for (int i = 0; i < C8NUM; i++) {
        tmp_buffer[i] = 0;
      }
      for (int kh = 0; kh < kernel_h; kh++) {
        const int8_t *src_kw = src_kh;
        const int16_t *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++) {
          for (int c = 0; c < C8NUM; c++) {
            tmp_buffer[c] += (src_kw[c] - in_zp[c]) * weight_kw[c];
          }
          src_kw += in_kw_step;
          weight_kw += C8NUM;
        }  // kernel_w loop
        src_kh += in_kh_step;
        weight_kh += kernel_w * C8NUM;
      }  // kernel_h loop
      // add bias relu
      for (int c = 0; c < C8NUM; c++) {
        tmp_buffer[c] += bias[c];
        tmp_buffer[c] = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(tmp_buffer[c] * (1 << (unsigned int)left_shift[c]), out_multiplier[c]),
          -right_shift[c]);
        tmp_buffer[c] += out_zp[c];
        tmp_buffer[c] = MSMAX(tmp_buffer[c], acc_min[c]);
        tmp_buffer[c] = MSMIN(tmp_buffer[c], acc_max[c]);
        dst_w[c] = (tmp_buffer[c]);
      }
      dst_w += block_channel;
      src_w += in_sw_step;
    }  // dst_width loop
    dst_h += out_h_step;
    src_h += in_sh_step;
  }  // dst_height loop
}
#endif

void ConvDwInt8SW(int8_t *output_data, const int8_t *input_data, const int16_t *weight_data, const int32_t *bias_data,
                  int8_t *input_zp, int32_t *output_zp, const ConvParameter *conv_param,
                  const SlidingWindowParam *sliding, int task_id) {
  const int8_t *src = input_data;
  int8_t *dst = output_data;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      const int8_t *src_data = src + oc * C8NUM;
      int8_t *dst_data = dst + oc * C8NUM;
      const int16_t *weight = weight_data + oc * sliding->kernel_step_;
      const int32_t *bias = bias_data + oc * C8NUM;

      int *out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_ + oc * C8NUM;
      int *left_shift = conv_param->conv_quant_arg_.left_shift_ + oc * C8NUM;
      int *right_shift = conv_param->conv_quant_arg_.right_shift_ + oc * C8NUM;
      int *acc_min = conv_param->conv_quant_arg_.out_act_min_ + oc * C8NUM;
      int *acc_max = conv_param->conv_quant_arg_.out_act_max_ + oc * C8NUM;
      int8_t *in_zp = input_zp + oc * C8NUM;
      int32_t *out_zp = output_zp + oc * C8NUM;

      ConvDwInt8Border(dst_data, src_data, weight, bias, 0, sliding->top_, 0, conv_param->output_w_, conv_param,
                       sliding, in_zp, out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max);
      ConvDwInt8Border(dst_data, src_data, weight, bias, sliding->bottom_, conv_param->output_h_, 0,
                       conv_param->output_w_, conv_param, sliding, in_zp, out_zp, out_multiplier, left_shift,
                       right_shift, acc_min, acc_max);
      ConvDwInt8Border(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, 0, sliding->left_, conv_param,
                       sliding, in_zp, out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max);
      ConvDwInt8Border(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, sliding->right_,
                       conv_param->output_w_, conv_param, sliding, in_zp, out_zp, out_multiplier, left_shift,
                       right_shift, acc_min, acc_max);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int in_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int in_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        const int8_t *in_t = src_data + in_h_start * sliding->in_h_step_ + in_w_start * sliding->block_channel_;
        int8_t *out_t = dst_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
        ConvDwInt8Center(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                         conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_, sliding->block_channel_,
                         sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_, in_zp,
                         out_zp, out_multiplier, left_shift, right_shift, acc_min, acc_max);
      }
    }  // output C8 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nhwc8
}
/*conv depthwise sliding window perchannel int8 end*/

/*deconv depthwise int8 begin*/
void DeconvDwInt8BorderPixel(int32_t *dst, const int16_t *src, const int16_t *weight, int height, int width,
                             int in_kh_step, int in_kw_step, int kernel_w) {
  int32_t *dst_kh = dst;
  const int16_t *weight_kh = weight;
  for (int kh = 0; kh < height; kh++) {
    int32_t *dst_kw = dst_kh;
    const int16_t *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
      for (int c = 0; c < C4NUM; c++) {
        dst_kw[c] += src[c] * weight_kw[c];
      }
      dst_kw += in_kw_step;
      weight_kw += C4NUM;
    }  // kernel_w loop
    dst_kh += in_kh_step;
    weight_kh += kernel_w * C4NUM;
  }  // kernel_h loop
}

void DeconvDwInt8Border(int32_t *dst, const int16_t *src, const int16_t *weight, int top, int bottom, int left,
                        int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  const int16_t *src_h = src + top * sliding->out_h_step_;
  for (int ih = top; ih < bottom; ih++) {
    int oh = ih * conv_param->stride_h_ - conv_param->pad_u_;
    int start_kh = MSMAX(0, UP_DIV(-oh, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->output_h_ - oh, conv_param->dilation_h_));
    int32_t *dst_h = dst + oh * sliding->in_h_step_;

    const int16_t *src_kernel = src_h + left * sliding->block_channel_;
    for (int iw = left; iw < right; iw++) {
      int ow = iw * conv_param->stride_w_ - conv_param->pad_l_;
      int start_kw = MSMAX(0, UP_DIV(-ow, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->output_w_ - ow, conv_param->dilation_w_));
      int32_t *dst_w = dst_h + ow * C4NUM;

      const int16_t *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C4NUM;
      int32_t *dst_kernel = dst_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;

      DeconvDwInt8BorderPixel(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                              sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_);
      src_kernel += sliding->block_channel_;
    }  // width loop
    src_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM
void DeconvDwInt8Center(int32_t *dst, const int16_t *src, const int16_t *weight, int height, int width, int kernel_h,
                        int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step, int in_kh_step,
                        int in_kw_step) {
  int32_t *dst_h = dst;
  const int16_t *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    int32_t *dst_w = dst_h;
    const int16_t *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      int32_t *dst_kh = dst_w;
      const int16_t *weight_kh = weight;
      for (int kh = 0; kh < kernel_h; kh++) {
        int32_t *dst_kw = dst_kh;
        const int16_t *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++) {
          for (int c = 0; c < C4NUM; c++) {
            dst_kw[c] += src_w[c] * weight_kw[c];
          }
          dst_kw += in_kw_step;
          weight_kw += C4NUM;
        }  // kernel_w loop
        dst_kh += in_kh_step;
        weight_kh += kernel_w * C4NUM;
      }  // kernel_h loop
      dst_w += in_sw_step;
      src_w += block_channel;
    }  // dst_width loop
    dst_h += in_sh_step;
    src_h += out_h_step;
  }  // dst_height loop
}
#endif

#ifndef ENABLE_ARM
void DeconvDwInt8Post(int8_t *dst, int32_t *output_buffer, const int32_t *bias, int block_channel, int pixel_nums,
                      int out_multiplier, int left_shift, int right_shift, int32_t out_zp, int32_t acc_min,
                      int32_t acc_max) {
  int8_t *dst_k = dst;
  int32_t *buffer_k = output_buffer;
  for (int k = 0; k < pixel_nums; k++) {
    for (int c = 0; c < C4NUM; c++) {
      buffer_k[c] += bias[c];
      buffer_k[c] = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(buffer_k[c] * (1 << (unsigned int)left_shift), out_multiplier), -right_shift);
      buffer_k[c] += out_zp;
      buffer_k[c] = MSMAX(buffer_k[c], acc_min);
      buffer_k[c] = MSMIN(buffer_k[c], acc_max);
      dst_k[c] = (buffer_k[c]);
    }
    dst_k += block_channel;
    buffer_k += C4NUM;
  }
}
#endif

void DeconvDwInt8(int8_t *output_data, int32_t *output_buffer, const int16_t *input_data, const int16_t *weight_data,
                  const int32_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                  int task_id) {
  const int16_t *src = input_data;
  int8_t *dst = output_data;
  int buffer_size = conv_param->output_h_ * conv_param->output_w_ * C4NUM;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      memset(output_buffer, 0, buffer_size * sizeof(int32_t));
      const int16_t *src_data = src + oc * C4NUM;
      const int16_t *weight = weight_data + oc * sliding->kernel_step_;
      const int32_t *bias = bias_data + oc * C4NUM;
      int8_t *dst_data = dst + oc * C4NUM;
      DeconvDwInt8Border(output_buffer, src_data, weight, 0, sliding->top_, 0, conv_param->input_w_, conv_param,
                         sliding);
      DeconvDwInt8Border(output_buffer, src_data, weight, sliding->bottom_, conv_param->input_h_, 0,
                         conv_param->input_w_, conv_param, sliding);
      DeconvDwInt8Border(output_buffer, src_data, weight, sliding->top_, sliding->bottom_, 0, sliding->left_,
                         conv_param, sliding);
      DeconvDwInt8Border(output_buffer, src_data, weight, sliding->top_, sliding->bottom_, sliding->right_,
                         conv_param->input_w_, conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int oh_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int oh_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        int32_t *out_t = output_buffer + oh_h_start * sliding->in_h_step_ + oh_w_start * sliding->block_channel_;
        const int16_t *in_t =
          src_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
#ifdef ENABLE_ARM
        DeconvDwInt8Center(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                           conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(int16_t),
                           sliding->block_channel_ * sizeof(int16_t), sliding->in_sh_step_ * sizeof(int32_t),
                           sliding->in_sw_step_ * sizeof(int32_t), sliding->in_kh_step_ * sizeof(int32_t),
                           sliding->in_kw_step_ * sizeof(int32_t));
#else
        DeconvDwInt8Center(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                           conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_, sliding->block_channel_,
                           sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_);
#endif
      }
      DeconvDwInt8Post(dst_data, output_buffer, bias, sliding->block_channel_,
                       conv_param->output_h_ * conv_param->output_w_, conv_param->conv_quant_arg_.quant_multiplier_[0],
                       conv_param->conv_quant_arg_.left_shift_[0], conv_param->conv_quant_arg_.right_shift_[0],
                       conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
                       conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0]);
    }  // output C4 loop
    src += sliding->out_step_;
    dst += sliding->in_step_;
  }  // batch loop
  // output nhwc4
}
/*deconv depthwise int8 end*/
