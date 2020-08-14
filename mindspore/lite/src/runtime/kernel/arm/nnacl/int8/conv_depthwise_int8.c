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
#include "nnacl/quantization/fixed_point.h"
#include "nnacl/int8/common_func.h"

/*conv depthwise int8 begin*/
void DepthwiseBorderPixelInt8(int8_t *dst, const int16_t *src, const int16_t *weight, const int32_t *bias, int height,
                              int width, int in_kh_step, int in_kw_step, int kernel_w, int *out_multiplier,
                              int *left_shift, int *right_shift, int32_t out_zp, int32_t acc_min, int32_t acc_max,
                              bool per_channel) {
  int tmp_buffer[C4NUM];
  for (int i = 0; i < C4NUM; i++) {
    tmp_buffer[i] = 0;
  }
  const int16_t *src_kh = src;
  const int16_t *weight_kh = weight;
  for (int kh = 0; kh < height; kh++) {
    const int16_t *src_kw = src_kh;
    const int16_t *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
      for (int c = 0; c < C4NUM; c++) {
        tmp_buffer[c] += src_kw[c] * weight_kw[c];
      }
      src_kw += in_kw_step;
      weight_kw += C4NUM;
    }  // kernel_w loop
    src_kh += in_kh_step;
    weight_kh += kernel_w * C4NUM;
  }  // kernel_h loop
  int32_t left = left_shift[0];
  int32_t right = right_shift[0];
  int32_t multiplier = out_multiplier[0];
  for (int c = 0; c < C4NUM; c++) {
    if (per_channel) {
      left = left_shift[c];
      right = right_shift[c];
      multiplier = out_multiplier[c];
    }
    tmp_buffer[c] += bias[c];
    tmp_buffer[c] = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(tmp_buffer[c] * (1 << (unsigned int)left), multiplier), right);
    tmp_buffer[c] += out_zp;
    tmp_buffer[c] = MSMAX(tmp_buffer[c], acc_min);
    tmp_buffer[c] = MSMIN(tmp_buffer[c], acc_max);
    dst[c] = (tmp_buffer[c]);
  }
}

void DepthwiseBorderInt8(int8_t *dst, const int16_t *src, const int16_t *weight, const int32_t *bias, int top,
                         int bottom, int left, int right, const ConvParameter *conv_param,
                         const SlidingWindowParam *sliding, int *out_multiplier, int *left_shift, int *right_shift,
                         bool per_channel) {
  int8_t *dst_h = dst + top * sliding->out_h_step_;
  for (int oh = top; oh < bottom; oh++) {
    int ih = oh * conv_param->stride_h_ - conv_param->pad_h_;
    int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));
    const int16_t *src_h = src + ih * sliding->in_h_step_;

    int8_t *dst_kernel = dst_h + left * sliding->block_channel_;
    for (int ow = left; ow < right; ow++) {
      int iw = ow * conv_param->stride_w_ - conv_param->pad_w_;
      int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));
      const int16_t *src_w = src_h + iw * sliding->block_channel_;

      const int16_t *src_kernel = src_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
      const int16_t *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C4NUM;

      DepthwiseBorderPixelInt8(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                               sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_, out_multiplier,
                               left_shift, right_shift, conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
                               conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0],
                               per_channel);

      dst_kernel += sliding->block_channel_;
    }  // width loop
    dst_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void DepthwiseCenterInt8(int8_t *dst, const int16_t *src, const int16_t *weight, const int32_t *bias, int height,
                         int width, int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step,
                         int in_sw_step, int in_kh_step, int in_kw_step, int *out_multiplier, int *left_shift,
                         int *right_shift, int32_t out_zp, int32_t acc_min, int32_t acc_max, bool per_channel) {
  int tmp_buffer[C4NUM];
  int8_t *dst_h = dst;
  const int16_t *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    int8_t *dst_w = dst_h;
    const int16_t *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      const int16_t *src_kh = src_w;
      const int16_t *weight_kh = weight;

      for (int i = 0; i < C4NUM; i++) {
        tmp_buffer[i] = 0;
      }
      for (int kh = 0; kh < kernel_h; kh++) {
        const int16_t *src_kw = src_kh;
        const int16_t *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++) {
          for (int c = 0; c < C4NUM; c++) {
            tmp_buffer[c] += src_kw[c] * weight_kw[c];
          }
          src_kw += in_kw_step;
          weight_kw += C4NUM;
        }  // kernel_w loop
        src_kh += in_kh_step;
        weight_kh += kernel_w * C4NUM;
      }  // kernel_h loop
      // add bias relu
      int32_t left = left_shift[0];
      int32_t right = right_shift[0];
      int32_t multiplier = out_multiplier[0];
      for (int c = 0; c < C4NUM; c++) {
        if (per_channel) {
          left = left_shift[c];
          right = right_shift[c];
          multiplier = out_multiplier[c];
        }
        tmp_buffer[c] += bias[c];
        tmp_buffer[c] = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(tmp_buffer[c] * (1 << (unsigned int)left), multiplier), -right);
        tmp_buffer[c] += out_zp;
        tmp_buffer[c] = MSMAX(tmp_buffer[c], acc_min);
        tmp_buffer[c] = MSMIN(tmp_buffer[c], acc_max);
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

void ConvDwInt8(int8_t *output_data, const int16_t *input_data, const int16_t *weight_data, const int32_t *bias_data,
                const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id) {
  const int16_t *src = input_data;
  int8_t *dst = output_data;
  bool per_channel = conv_param->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL;
  int *out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_;
  int *left_shift = conv_param->conv_quant_arg_.left_shift_;
  int *right_shift = conv_param->conv_quant_arg_.right_shift_;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      const int16_t *src_data = src + oc * C4NUM;
      int8_t *dst_data = dst + oc * C4NUM;
      const int16_t *weight = weight_data + oc * sliding->kernel_step_;
      const int32_t *bias = bias_data + oc * C4NUM;

      if (per_channel) {
        out_multiplier = conv_param->conv_quant_arg_.quant_multiplier_ + oc;
        left_shift = conv_param->conv_quant_arg_.left_shift_ + oc;
        right_shift = conv_param->conv_quant_arg_.right_shift_ + oc;
      }

      DepthwiseBorderInt8(dst_data, src_data, weight, bias, 0, sliding->top_, 0, conv_param->output_w_, conv_param,
                          sliding, out_multiplier, left_shift, right_shift, per_channel);
      DepthwiseBorderInt8(dst_data, src_data, weight, bias, sliding->bottom_, conv_param->output_h_, 0,
                          conv_param->output_w_, conv_param, sliding, out_multiplier, left_shift, right_shift,
                          per_channel);
      DepthwiseBorderInt8(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, 0, sliding->left_,
                          conv_param, sliding, out_multiplier, left_shift, right_shift, per_channel);
      DepthwiseBorderInt8(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, sliding->right_,
                          conv_param->output_w_, conv_param, sliding, out_multiplier, left_shift, right_shift,
                          per_channel);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int in_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_h_;
        int in_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_w_;
        const int16_t *in_t = src_data + in_h_start * sliding->in_h_step_ + in_w_start * sliding->block_channel_;
        int8_t *out_t = dst_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
#ifdef ENABLE_ARM64
        ConvDwInt8Center(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                         conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(int8_t),
                         sliding->block_channel_ * sizeof(int8_t), sliding->in_sh_step_ * sizeof(int16_t),
                         sliding->in_sw_step_ * sizeof(int16_t), sliding->in_kh_step_ * sizeof(int16_t),
                         sliding->in_kw_step_ * sizeof(int16_t), conv_param->conv_quant_arg_.quant_multiplier_[0],
                         conv_param->conv_quant_arg_.left_shift_[0], conv_param->conv_quant_arg_.right_shift_[0],
                         conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
                         conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0]);
#else

        DepthwiseCenterInt8(
          out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
          conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_, sliding->block_channel_,
          sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_, out_multiplier,
          left_shift, right_shift, conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
          conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0], per_channel);
#endif
      }
    }  // output C4 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nhwc4
}
/*conv depthwise int8 end*/

/*deconv depthwise int8 begin*/
void DeconvDepthwiseBorderPixelInt8(int32_t *dst, const int16_t *src, const int16_t *weight, int height, int width,
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

void DeconvDepthwiseBorderInt8(int32_t *dst, const int16_t *src, const int16_t *weight, int top, int bottom, int left,
                               int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  const int16_t *src_h = src + top * sliding->out_h_step_;
  for (int ih = top; ih < bottom; ih++) {
    int oh = ih * conv_param->stride_h_ - conv_param->pad_h_;
    int start_kh = MSMAX(0, UP_DIV(-oh, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->output_h_ - oh, conv_param->dilation_h_));
    int32_t *dst_h = dst + oh * sliding->in_h_step_;

    const int16_t *src_kernel = src_h + left * sliding->block_channel_;
    for (int iw = left; iw < right; iw++) {
      int ow = iw * conv_param->stride_w_ - conv_param->pad_w_;
      int start_kw = MSMAX(0, UP_DIV(-ow, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->output_w_ - ow, conv_param->dilation_w_));
      int32_t *dst_w = dst_h + ow * C4NUM;

      const int16_t *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C4NUM;
      int32_t *dst_kernel = dst_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;

      DeconvDepthwiseBorderPixelInt8(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                                     sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_);
      src_kernel += sliding->block_channel_;
    }  // width loop
    src_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void DeconvDepthwiseCenterInt8(int32_t *dst, const int16_t *src, const int16_t *weight, int height, int width,
                               int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step,
                               int in_sw_step, int in_kh_step, int in_kw_step) {
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

void DeconvDepthwisePostFuncInt8(int8_t *dst, int32_t *output_buffer, const int32_t *bias, int block_channel,
                                 const ConvParameter *conv_param, int out_multiplier, int left_shift, int right_shift,
                                 int32_t out_zp, int32_t acc_min, int32_t acc_max) {
  int8_t *dst_k = dst;
  int32_t *buffer_k = output_buffer;
  for (int k = 0; k < conv_param->output_h_ * conv_param->output_w_; k++) {
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
      DeconvDepthwiseBorderInt8(output_buffer, src_data, weight, 0, sliding->top_, 0, conv_param->input_w_, conv_param,
                                sliding);
      DeconvDepthwiseBorderInt8(output_buffer, src_data, weight, sliding->bottom_, conv_param->input_h_, 0,
                                conv_param->input_w_, conv_param, sliding);
      DeconvDepthwiseBorderInt8(output_buffer, src_data, weight, sliding->top_, sliding->bottom_, 0, sliding->left_,
                                conv_param, sliding);
      DeconvDepthwiseBorderInt8(output_buffer, src_data, weight, sliding->top_, sliding->bottom_, sliding->right_,
                                conv_param->input_w_, conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int oh_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_h_;
        int oh_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_w_;
        int32_t *out_t = output_buffer + oh_h_start * sliding->in_h_step_ + oh_w_start * sliding->block_channel_;
        const int16_t *in_t =
          src_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
#ifdef ENABLE_ARM64
        DeconvDwInt8Center(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                           conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(int16_t),
                           sliding->block_channel_ * sizeof(int16_t), sliding->in_sh_step_ * sizeof(int32_t),
                           sliding->in_sw_step_ * sizeof(int32_t), sliding->in_kh_step_ * sizeof(int32_t),
                           sliding->in_kw_step_ * sizeof(int32_t));
#else
        DeconvDepthwiseCenterInt8(out_t, in_t, weight, sliding->bottom_ - sliding->top_,
                                  sliding->right_ - sliding->left_, conv_param->kernel_h_, conv_param->kernel_w_,
                                  sliding->out_h_step_, sliding->block_channel_, sliding->in_sh_step_,
                                  sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_);
#endif
      }
      DeconvDepthwisePostFuncInt8(
        dst_data, output_buffer, bias, sliding->block_channel_, conv_param,
        conv_param->conv_quant_arg_.quant_multiplier_[0], conv_param->conv_quant_arg_.left_shift_[0],
        conv_param->conv_quant_arg_.right_shift_[0], conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
        conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0]);
    }  // output C4 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nhwc4
}
/*deconv depthwise int8 end*/
