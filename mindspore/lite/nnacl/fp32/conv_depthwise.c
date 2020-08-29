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

#include "nnacl/fp32/conv_depthwise.h"
#include "nnacl/fp32/common_func.h"
#include "nnacl/winograd_transform.h"
#ifdef ENABLE_ARM64
#include <arm_neon.h>
#endif

#ifndef ENABLE_ARM64
void ConvDwFp32Row(float *output_ptr, const float *input_ptr, const float *weight_ptr, int num_pixels,
                   int output_channel, int input_step) {
  for (int i = 0; i < num_pixels; i++) {
    for (int c = 0; c < output_channel; c++) {
      *output_ptr++ += weight_ptr[c] * input_ptr[c];
    }
    input_ptr += input_step;
  }
}
#endif

void ConvDw(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
            const ConvParameter *conv_param, int task_id) {
  int h_step = UP_DIV(conv_param->output_h_, conv_param->thread_num_);
  int h_start = h_step * task_id;
  int h_end = MSMIN(h_start + h_step, conv_param->output_h_);
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    const float *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    float *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    for (int oh = h_start; oh < h_end; oh++) {
      float *dst_data = dst + oh * conv_param->output_w_ * conv_param->output_channel_;

      int ih_origin = oh * conv_param->stride_h_ - conv_param->pad_u_;
      int start_kh = MSMAX(0, UP_DIV(-ih_origin, conv_param->dilation_h_));
      int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih_origin, conv_param->dilation_h_));

      for (int ow = 0; ow < conv_param->output_w_; ow++) {
        memcpy(dst_data + ow * conv_param->output_channel_, bias_data, conv_param->output_channel_ * sizeof(float));
      }
      for (int kh = start_kh; kh < end_kh; kh++) {
        int ih = ih_origin + conv_param->dilation_w_ * kh;

        const float *src_kh = src + ih * conv_param->input_w_ * conv_param->input_channel_;
        const float *weight_kh = weight_data + kh * conv_param->kernel_w_ * conv_param->output_channel_;

        int in_sw_step = conv_param->stride_w_ * conv_param->input_channel_;
        for (int kw = 0; kw < conv_param->kernel_w_; kw++) {
          int out_w_start = MSMAX(
            0, (conv_param->pad_l_ - conv_param->dilation_w_ * kw + conv_param->stride_w_ - 1) / conv_param->stride_w_);
          int out_w_end = MSMIN(conv_param->output_w_, (conv_param->input_w_ + conv_param->pad_l_ -
                                                        conv_param->dilation_w_ * kw + conv_param->stride_w_ - 1) /
                                                         conv_param->stride_w_);

          float *dst_w = dst_data + out_w_start * conv_param->output_channel_;
          int iw_origin = (out_w_start * conv_param->stride_w_) - conv_param->pad_l_ + conv_param->dilation_w_ * kw;

          const float *src_kw = src_kh + iw_origin * conv_param->input_channel_;
          int num_pixels = out_w_end - out_w_start;

          ConvDwFp32Row(dst_w, src_kw, weight_kh, num_pixels, conv_param->output_channel_, in_sw_step);
          weight_kh += conv_param->output_channel_;
        }
      }
      if (relu) {
        ReluFp32(dst_data, dst_data, conv_param->output_w_ * conv_param->output_channel_);
      }
      if (relu6) {
        Relu6Fp32(dst_data, dst_data, conv_param->output_w_ * conv_param->output_channel_);
      }
    }
  }
}

void InitSlidingParam(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block) {
  int left = 0;
  int right = conv_param->output_w_;
  int top = 0;
  int bottom = conv_param->output_h_;

  for (; left * conv_param->stride_w_ < conv_param->pad_l_; left++) {
  }
  for (; (right - 1) * conv_param->stride_w_ - conv_param->pad_l_ + conv_param->kernel_w_ * conv_param->dilation_w_ >
           conv_param->input_w_ &&
         right > left;
       right--) {
  }
  for (; top * conv_param->stride_h_ < conv_param->pad_u_; top++) {
  }
  for (; (bottom - 1) * conv_param->stride_h_ - conv_param->pad_u_ + conv_param->kernel_h_ * conv_param->dilation_h_ >
           conv_param->input_h_ &&
         bottom > top;
       bottom--) {
  }
  sliding->left_ = left;
  sliding->right_ = right;
  sliding->top_ = top;
  sliding->bottom_ = bottom;
  sliding->c_block_ = UP_DIV(conv_param->output_channel_, block);
  sliding->block_channel_ = UP_DIV(conv_param->output_channel_, block) * block;
  sliding->out_step_ = conv_param->output_h_ * conv_param->output_w_ * sliding->block_channel_;
  sliding->out_h_step_ = conv_param->output_w_ * sliding->block_channel_;
}

void InitSlidingParamConv(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block) {
  InitSlidingParam(sliding, conv_param, block);
  AppendSlidingParamConv(sliding, conv_param, block);
}

void AppendSlidingParamConv(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block) {
  int in_channel = conv_param->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int ic4_channel = ic4 * C4NUM;
  sliding->ic4_channel_ = ic4_channel;
  sliding->in_step_ = conv_param->input_h_ * conv_param->input_w_ * ic4_channel;  // for batch loop
  sliding->in_h_step_ = conv_param->input_w_ * ic4_channel;
  sliding->in_sh_step_ = conv_param->input_w_ * ic4_channel * conv_param->stride_h_;    // stride H
  sliding->in_sw_step_ = ic4_channel * conv_param->stride_w_;                           // stride W
  sliding->in_kh_step_ = conv_param->input_w_ * ic4_channel * conv_param->dilation_h_;  // kernel H
  sliding->in_kw_step_ = ic4_channel * conv_param->dilation_w_;                         // kernel W
  sliding->kernel_step_ = conv_param->kernel_w_ * conv_param->kernel_h_ * ic4_channel * block;
}

void InitSlidingParamConvDw(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block) {
  InitSlidingParam(sliding, conv_param, block);
  AppendSlidingParamConvDw(sliding, conv_param, block);
}

void AppendSlidingParamConvDw(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block) {
  sliding->in_step_ = conv_param->input_h_ * conv_param->input_w_ * sliding->block_channel_;  // for batch loop
  sliding->in_h_step_ = conv_param->input_w_ * sliding->block_channel_;
  sliding->in_sh_step_ = conv_param->input_w_ * sliding->block_channel_ * conv_param->stride_h_;    // stride H
  sliding->in_sw_step_ = sliding->block_channel_ * conv_param->stride_w_;                           // stride W
  sliding->in_kh_step_ = conv_param->input_w_ * sliding->block_channel_ * conv_param->dilation_h_;  // kernel H
  sliding->in_kw_step_ = sliding->block_channel_ * conv_param->dilation_w_;                         // kernel W
  sliding->kernel_step_ = conv_param->kernel_w_ * conv_param->kernel_h_ * block;
}

/*conv depthwise fp32 begin*/
#ifndef ENABLE_ARM64
void DepthwiseBorderPixel(float *dst, const float *src, const float *weight, const float *bias, int height, int width,
                          int in_kh_step, int in_kw_step, int kernel_w_step, bool is_relu, bool is_relu6) {
  const float *src_kh = src;
  const float *weight_kh = weight;
  for (int c = 0; c < C4NUM; c++) {
    dst[c] = 0;
  }
  for (int kh = 0; kh < height; kh++) {
    const float *src_kw = src_kh;
    const float *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
      for (int c = 0; c < C4NUM; c++) {
        dst[c] += src_kw[c] * weight_kw[c];
      }
      src_kw += in_kw_step;
      weight_kw += C4NUM;
    }  // kernel_w loop
    src_kh += in_kh_step;
    weight_kh += kernel_w_step;
  }  // kernel_h loop
  for (int c = 0; c < C4NUM; c++) {
    dst[c] += bias[c];
    dst[c] = (is_relu) ? (MSMAX(0, dst[c])) : (dst[c]);
    dst[c] = (is_relu6) ? (MSMIN(6, MSMAX(0, dst[c]))) : (dst[c]);
  }
}
#endif

void DepthwiseBorder(float *dst, const float *src, const float *weight, const float *bias, int top, int bottom,
                     int left, int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  float *dst_h = dst + top * sliding->out_h_step_;
  for (int oh = top; oh < bottom; oh++) {
    int ih = oh * conv_param->stride_h_ - conv_param->pad_u_;
    int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));
    const float *src_h = src + ih * sliding->in_h_step_;

    float *dst_kernel = dst_h + left * sliding->block_channel_;
    for (int ow = left; ow < right; ow++) {
      int iw = ow * conv_param->stride_w_ - conv_param->pad_l_;
      int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));
      const float *src_w = src_h + iw * sliding->block_channel_;

      const float *src_kernel = src_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
      const float *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C4NUM;

#ifdef ENABLE_ARM64
      ConvDwFp32Border(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                       sliding->in_kh_step_ * sizeof(float), sliding->in_kw_step_ * sizeof(float),
                       conv_param->kernel_w_ * C4NUM * sizeof(float), relu, relu6);
#else
      DepthwiseBorderPixel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                           sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_ * C4NUM, relu, relu6);
#endif
      dst_kernel += sliding->block_channel_;
    }  // width loop
    dst_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void DepthwiseCenter(float *dst, const float *src, const float *weight, const float *bias, int height, int width,
                     int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step,
                     int in_kh_step, int in_kw_step, bool is_relu, bool is_relu6) {
  float *dst_h = dst;
  const float *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float *dst_w = dst_h;
    const float *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      const float *src_kh = src_w;
      const float *weight_kh = weight;
      for (int c = 0; c < C4NUM; c++) {
        dst_w[c] = 0;
      }
      for (int kh = 0; kh < kernel_h; kh++) {
        const float *src_kw = src_kh;
        const float *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++) {
          for (int c = 0; c < C4NUM; c++) {
            dst_w[c] += src_kw[c] * weight_kw[c];
          }
          src_kw += in_kw_step;
          weight_kw += C4NUM;
        }  // kernel_w loop
        src_kh += in_kh_step;
        weight_kh += kernel_w * C4NUM;
      }  // kernel_h loop
      // add biad relu
      for (int c = 0; c < C4NUM; c++) {
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

// conv depthwise fp32: sliding window
void ConvDwC4Fp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                  const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id) {
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  const float *src = input_data;
  float *dst = output_data;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      const float *src_data = src + oc * C4NUM;
      float *dst_data = dst + oc * C4NUM;
      const float *weight = weight_data + oc * sliding->kernel_step_;
      const float *bias = bias_data + oc * C4NUM;
      DepthwiseBorder(dst_data, src_data, weight, bias, 0, sliding->top_, 0, conv_param->output_w_, conv_param,
                      sliding);
      DepthwiseBorder(dst_data, src_data, weight, bias, sliding->bottom_, conv_param->output_h_, 0,
                      conv_param->output_w_, conv_param, sliding);
      DepthwiseBorder(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, 0, sliding->left_, conv_param,
                      sliding);
      DepthwiseBorder(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, sliding->right_,
                      conv_param->output_w_, conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int in_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int in_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        const float *in_t = src_data + in_h_start * sliding->in_h_step_ + in_w_start * sliding->block_channel_;
        float *out_t = dst_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
#ifdef ENABLE_ARM64
        ConvDwFp32Center(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                         conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(float),
                         sliding->block_channel_ * sizeof(float), sliding->in_sh_step_ * sizeof(float),
                         sliding->in_sw_step_ * sizeof(float), sliding->in_kh_step_ * sizeof(float),
                         sliding->in_kw_step_ * sizeof(float), relu, relu6);
#else
        DepthwiseCenter(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                        conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_, sliding->block_channel_,
                        sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_, relu,
                        relu6);
#endif
      }
    }  // output C4 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nhwc4
}
/*conv depthwise fp32 end*/

/*deconv depthwise fp32 begin*/
void DeconvDepthwiseBorderPixel(float *dst, const float *src, const float *weight, int height, int width,
                                int in_kh_step, int in_kw_step, int kernel_w_step) {
  float *dst_kh = dst;
  const float *weight_kh = weight;
  for (int kh = 0; kh < height; kh++) {
    float *dst_kw = dst_kh;
    const float *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
#ifdef ENABLE_ARM64
      float32x4_t src_4 = vld1q_f32(src);
      float32x4_t weight_4 = vld1q_f32(weight_kw);
      float32x4_t dst_4 = vld1q_f32(dst_kw);
      dst_4 = vfmaq_f32(dst_4, src_4, weight_4);
      vst1q_f32(dst_kw, dst_4);
#else
      for (int c = 0; c < C4NUM; c++) {
        dst_kw[c] += src[c] * weight_kw[c];
      }
#endif
      dst_kw += in_kw_step;
      weight_kw += C4NUM;
    }  // kernel_w loop
    dst_kh += in_kh_step;
    weight_kh += kernel_w_step;
  }  // kernel_h loop
}

void DeconvDepthwiseBorder(float *dst, const float *src, const float *weight, int top, int bottom, int left, int right,
                           const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  const float *src_h = src + top * sliding->out_h_step_;
  for (int ih = top; ih < bottom; ih++) {
    int oh = ih * conv_param->stride_h_ - conv_param->pad_u_;
    int start_kh = MSMAX(0, UP_DIV(-oh, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->output_h_ - oh, conv_param->dilation_h_));
    float *dst_h = dst + oh * sliding->in_h_step_;

    const float *src_kernel = src_h + left * sliding->block_channel_;
    for (int iw = left; iw < right; iw++) {
      int ow = iw * conv_param->stride_w_ - conv_param->pad_l_;
      int start_kw = MSMAX(0, UP_DIV(-ow, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->output_w_ - ow, conv_param->dilation_w_));
      float *dst_w = dst_h + ow * sliding->block_channel_;

      const float *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C4NUM;
      float *dst_kernel = dst_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
#ifdef ENABLE_ARM64
      DeconvDwFp32Border(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                         sliding->in_kh_step_ * sizeof(float), sliding->in_kw_step_ * sizeof(float),
                         conv_param->kernel_w_ * C4NUM * sizeof(float));
#else
      DeconvDepthwiseBorderPixel(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                                 sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_ * C4NUM);
#endif
      src_kernel += sliding->block_channel_;
    }  // width loop
    src_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void DeconvDepthwiseCenter(float *dst, const float *src, const float *weight, int height, int width, int kernel_h,
                           int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step,
                           int in_kh_step, int in_kw_step) {
  float *dst_h = dst;
  const float *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float *dst_w = dst_h;
    const float *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      float *dst_kh = dst_w;
      const float *weight_kh = weight;
      for (int kh = 0; kh < kernel_h; kh++) {
        float *dst_kw = dst_kh;
        const float *weight_kw = weight_kh;
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

void DeconvDepthwisePostFunc(float *dst, const float *bias, int block_channel, const ConvParameter *conv_param) {
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;
  float *dst_k = dst;
  for (int k = 0; k < conv_param->output_h_ * conv_param->output_w_; k++) {
    for (int c = 0; c < C4NUM; c++) {
      dst_k[c] += bias[c];
      dst_k[c] = (relu) ? (MSMAX(0, dst_k[c])) : (dst_k[c]);
      dst_k[c] = (relu6) ? (MSMIN(6, MSMAX(0, dst_k[c]))) : (dst_k[c]);
    }
    dst_k += block_channel;
  }
}

// deconv depthwise fp32: sliding window
void DeconvDwC4Fp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                    const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id) {
  const float *src = input_data;
  float *dst = output_data;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      const float *src_data = src + oc * C4NUM;
      float *dst_data = dst + oc * C4NUM;
      const float *weight = weight_data + oc * sliding->kernel_step_;
      const float *bias = bias_data + oc * C4NUM;
      DeconvDepthwiseBorder(dst_data, src_data, weight, 0, sliding->top_, 0, conv_param->input_w_, conv_param, sliding);
      DeconvDepthwiseBorder(dst_data, src_data, weight, sliding->bottom_, conv_param->input_h_, 0, conv_param->input_w_,
                            conv_param, sliding);
      DeconvDepthwiseBorder(dst_data, src_data, weight, sliding->top_, sliding->bottom_, 0, sliding->left_, conv_param,
                            sliding);
      DeconvDepthwiseBorder(dst_data, src_data, weight, sliding->top_, sliding->bottom_, sliding->right_,
                            conv_param->input_w_, conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int oh_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int oh_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        float *out_t = dst_data + oh_h_start * sliding->in_h_step_ + oh_w_start * sliding->block_channel_;
        const float *in_t = src_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;

#ifdef ENABLE_ARM64
        DeconvDwFp32Center(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                           conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(float),
                           sliding->block_channel_ * sizeof(float), sliding->in_sh_step_ * sizeof(float),
                           sliding->in_sw_step_ * sizeof(float), sliding->in_kh_step_ * sizeof(float),
                           sliding->in_kw_step_ * sizeof(float));
#else
        DeconvDepthwiseCenter(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                              conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_,
                              sliding->block_channel_, sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_,
                              sliding->in_kw_step_);
#endif
      }
      DeconvDepthwisePostFunc(dst_data, bias, sliding->block_channel_, conv_param);
    }  // output C4 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nhwc4
}
/*deconv depthwise fp32 end*/
