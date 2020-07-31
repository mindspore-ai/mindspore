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

#include "src/runtime/kernel/arm/opclib/fp32/conv_depthwise.h"
#include "src/runtime/kernel/arm/opclib/fp32/common_func.h"
#ifdef ENABLE_ARM64
#include <arm_neon.h>
#endif

void InitSlidingParam(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block) {
  int left = 0;
  int right = conv_param->output_w_;
  int top = 0;
  int bottom = conv_param->output_h_;

  for (; left * conv_param->stride_w_ < conv_param->pad_w_; left++) {
  }
  for (; (right - 1) * conv_param->stride_w_ - conv_param->pad_w_ + conv_param->kernel_w_ * conv_param->dilation_w_ >
           conv_param->input_w_ &&
         right > left;
       right--) {
  }
  for (; top * conv_param->stride_h_ < conv_param->pad_h_; top++) {
  }
  for (; (bottom - 1) * conv_param->stride_h_ - conv_param->pad_h_ + conv_param->kernel_h_ * conv_param->dilation_h_ >
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
  sliding->in_step_ = conv_param->input_h_ * conv_param->input_w_ * sliding->block_channel_;  // for batch loop
  sliding->in_h_step_ = conv_param->input_w_ * sliding->block_channel_;
  sliding->in_sh_step_ = conv_param->input_w_ * sliding->block_channel_ * conv_param->stride_h_;    // stride H
  sliding->in_sw_step_ = sliding->block_channel_ * conv_param->stride_h_;                           // stride W
  sliding->in_kh_step_ = conv_param->input_w_ * sliding->block_channel_ * conv_param->dilation_h_;  // kernel H
  sliding->in_kw_step_ = sliding->block_channel_ * conv_param->dilation_w_;                         // kernel W
  sliding->kernel_step_ = conv_param->kernel_w_ * conv_param->kernel_h_ * block;
}

/*conv depthwise fp32 begin*/
void DepthwiseBorderPixel(float *dst, const float *src, const float *weight, const float *bias, int height, int width,
                          int in_kh_step, int in_kw_step, int kernel_w, bool is_relu, bool is_relu6) {
  const float *src_kh = src;
  const float *weight_kh = weight;
  for (int kh = 0; kh < height; kh++) {
    const float *src_kw = src_kh;
    const float *weight_kw = weight_kh;
    for (int kw = 0; kw < width; kw++) {
#ifdef ENABLE_ARM64
      float32x4_t src_4 = vld1q_f32(src_kw);
      float32x4_t weight_4 = vld1q_f32(weight_kw);
      float32x4_t dst_4 = vld1q_f32(dst);
      dst_4 = vfmaq_f32(dst_4, src_4, weight_4);
      vst1q_f32(dst, dst_4);
#else
      for (int c = 0; c < C4NUM; c++) {
        dst[c] += src_kw[c] * weight_kw[c];
      }
#endif
      src_kw += in_kw_step;
      weight_kw += C4NUM;
    }  // kernel_w loop
    src_kh += in_kh_step;
    weight_kh += kernel_w * C4NUM;
  }  // kernel_h loop
  for (int c = 0; c < C4NUM; c++) {
    dst[c] += bias[c];
    dst[c] = (is_relu) ? (MSMAX(0, dst[c])) : (dst[c]);
    dst[c] = (is_relu6) ? (MSMIN(6, MSMAX(0, dst[c]))) : (dst[c]);
  }
}

void DepthwiseBorder(float *dst, const float *src, const float *weight, const float *bias, int top, int bottom,
                     int left, int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  float *dst_h = dst + top * sliding->out_h_step_;
  for (int oh = top; oh < bottom; oh++) {
    int ih = oh * conv_param->stride_h_ - conv_param->pad_h_;
    int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));
    const float *src_h = src + ih * sliding->in_h_step_;

    float *dst_kernel = dst_h + left * sliding->block_channel_;
    for (int ow = left; ow < right; ow++) {
      int iw = ow * conv_param->stride_w_ - conv_param->pad_w_;
      int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));
      const float *src_w = src_h + iw * sliding->block_channel_;

      const float *src_kernel = src_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;
      const float *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C4NUM;

      DepthwiseBorderPixel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                           sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_, conv_param->is_relu_,
                           conv_param->is_relu6_);

      dst_kernel += sliding->block_channel_;
    }  // width loop
    dst_h += sliding->out_h_step_;
  }  // height loop
}

void DepthwiseCenter(float *dst, const float *src, const float *weight, const float *bias, int height, int width,
                     int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step,
                     int in_kh_step, int in_kw_step, bool is_relu, bool is_relu6) {
#ifdef ENABLE_ARM64
  ConvDwFp32Center(dst, src, weight, bias, height, width, kernel_h, kernel_w, out_h_step, block_channel,
                   in_sh_step, in_sw_step, in_kh_step, in_kw_step, is_relu, is_relu6);
#else
  float *dst_h = dst;
  const float *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float *dst_w = dst_h;
    const float *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      const float *src_kh = src_w;
      const float *weight_kh = weight;
      for (int kh = 0; kh < kernel_h; kh++) {
        const float *src_kw = src_kh;
        const float *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++) {
#ifdef ENABLE_ARM64
          float32x4_t src_4 = vld1q_f32(src_kw);
          float32x4_t weight_4 = vld1q_f32(weight_kw);
          float32x4_t dst_4 = vld1q_f32(dst_w);
          dst_4 = vfmaq_f32(dst_4, src_4, weight_4);
          vst1q_f32(dst_w, dst_4);
#else
          for (int c = 0; c < C4NUM; c++) {
            dst_w[c] += src_kw[c] * weight_kw[c];
          }
#endif
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
#endif
}

// conv depthwise fp32: sliding window
void ConvDwC4Fp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                  const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id) {
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
        int in_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_h_;
        int in_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_w_;
        const float *in_t = src_data + in_h_start * sliding->in_h_step_ + in_w_start * sliding->block_channel_;
        float *out_t = dst_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;

        DepthwiseCenter(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                        conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_, sliding->block_channel_,
                        sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_,
                        conv_param->is_relu_, conv_param->is_relu6_);
      }
    }  // output C4 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nc4hwc4
}
/*conv depthwise fp32 end*/

/*deconv depthwise fp32 begin*/
void DeconvDepthwiseBorderPixel(float *dst, const float *src, const float *weight, int height, int width,
                                int in_kh_step, int in_kw_step, int kernel_w) {
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
    weight_kh += kernel_w * C4NUM;
  }  // kernel_h loop
}

void DeconvDepthwiseBorder(float *dst, const float *src, const float *weight, int top, int bottom, int left, int right,
                           const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
  const float *src_h = src + top * sliding->out_h_step_;
  for (int ih = top; ih < bottom; ih++) {
    int oh = ih * conv_param->stride_h_ - conv_param->pad_h_;
    int start_kh = MSMAX(0, UP_DIV(-oh, conv_param->dilation_h_));
    int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->output_h_ - oh, conv_param->dilation_h_));
    float *dst_h = dst + oh * sliding->in_h_step_;

    const float *src_kernel = src_h + left * sliding->block_channel_;
    for (int iw = left; iw < right; iw++) {
      int ow = iw * conv_param->stride_w_ - conv_param->pad_w_;
      int start_kw = MSMAX(0, UP_DIV(-ow, conv_param->dilation_w_));
      int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->output_w_ - ow, conv_param->dilation_w_));
      float *dst_w = dst_h + ow * sliding->block_channel_;

      const float *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C4NUM;
      float *dst_kernel = dst_w + start_kh * sliding->in_kh_step_ + start_kw * sliding->in_kw_step_;

      DeconvDepthwiseBorderPixel(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                                 sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_);
      src_kernel += sliding->block_channel_;
    }  // width loop
    src_h += sliding->out_h_step_;
  }  // height loop
}

void DeconvDepthwiseCenter(float *dst, const float *src, const float *weight, int height, int width, int kernel_h,
                           int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step,
                           int in_kh_step, int in_kw_step) {
#ifdef ENABLE_ARM64
  DeconvDwFp32Center(dst, src, weight, height, width, kernel_h, kernel_w, out_h_step, block_channel,
                     in_sh_step, in_sw_step, in_kh_step, in_kw_step);
#else
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
#ifdef ENABLE_ARM64
          float32x4_t src_4 = vld1q_f32(src_w);
          float32x4_t weight_4 = vld1q_f32(weight_kw);
          float32x4_t dst_4 = vld1q_f32(dst_kw);
          dst_4 = vfmaq_f32(dst_4, src_4, weight_4);
          vst1q_f32(dst_kw, dst_4);
#else
          for (int c = 0; c < C4NUM; c++) {
            dst_kw[c] += src_w[c] * weight_kw[c];
          }
#endif
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
#endif
}

void DeconvDepthwisePostFunc(float *dst, const float *bias, int block_channel, const ConvParameter *conv_param) {
  float *dst_k = dst;
  for (int k = 0; k < conv_param->output_h_ * conv_param->output_w_; k++) {
    for (int c = 0; c < C4NUM; c++) {
      dst_k[c] += bias[c];
      dst_k[c] = (conv_param->is_relu_) ? (MSMAX(0, dst_k[c])) : (dst_k[c]);
      dst_k[c] = (conv_param->is_relu6_) ? (MSMIN(6, MSMAX(0, dst_k[c]))) : (dst_k[c]);
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
        int oh_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_h_;
        int oh_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_w_;
        float *out_t = dst_data + oh_h_start * sliding->in_h_step_ + oh_w_start * sliding->block_channel_;
        const float *in_t = src_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;

        DeconvDepthwiseCenter(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                              conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_,
                              sliding->block_channel_, sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_,
                              sliding->in_kw_step_);
      }
      DeconvDepthwisePostFunc(dst_data, bias, sliding->block_channel_, conv_param);
    }  // output C4 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nc4hwc4
}
/*deconv depthwise fp32 end*/
