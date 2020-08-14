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

#ifdef ENABLE_ARM64
      ConvDwFp32Border(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                       sliding->in_kh_step_ * sizeof(float), sliding->in_kw_step_ * sizeof(float),
                       conv_param->kernel_w_ * C4NUM * sizeof(float), conv_param->is_relu_, conv_param->is_relu6_);
#else
      DepthwiseBorderPixel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                           sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_ * C4NUM,
                           conv_param->is_relu_, conv_param->is_relu6_);
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
#ifdef ENABLE_ARM64
        ConvDwFp32Center(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                         conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(float),
                         sliding->block_channel_ * sizeof(float), sliding->in_sh_step_ * sizeof(float),
                         sliding->in_sw_step_ * sizeof(float), sliding->in_kh_step_ * sizeof(float),
                         sliding->in_kw_step_ * sizeof(float), conv_param->is_relu_, conv_param->is_relu6_);
#else
        DepthwiseCenter(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                        conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_, sliding->block_channel_,
                        sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_,
                        conv_param->is_relu_, conv_param->is_relu6_);
#endif
      }
    }  // output C4 loop
    src += sliding->in_step_;
    dst += sliding->out_step_;
  }  // batch loop
  // output nhwc4
}
/*conv depthwise fp32 end*/

/*conv depthwise 3x3 fp32 begin*/
void ConvDw3x3Fp32FilterTrans(float *trans_weight, float *weight, int oc4) {
  for (int c = 0; c < oc4; c++) {
    float *src = weight + c * C4NUM * 9;
    float *dst = trans_weight + c * C4NUM * 16;
#ifdef ENABLE_ARM
    float32x4_t g00 = vld1q_f32(src);
    float32x4_t g01 = vld1q_f32(src + 4);
    float32x4_t g02 = vld1q_f32(src + 2 * 4);
    float32x4_t g10 = vld1q_f32(src + 3 * 4);
    float32x4_t g11 = vld1q_f32(src + 4 * 4);
    float32x4_t g12 = vld1q_f32(src + 5 * 4);
    float32x4_t g20 = vld1q_f32(src + 6 * 4);
    float32x4_t g21 = vld1q_f32(src + 7 * 4);
    float32x4_t g22 = vld1q_f32(src + 8 * 4);

    float32x4_t dst00 = g00;
    float32x4_t dst01 = g01;
    float32x4_t dst02 = g02;

    float32x4_t dst10 = vaddq_f32(vmulq_n_f32(g00, 0.5), vmulq_n_f32(g10, 0.5));
    dst10 = vaddq_f32(dst10, vmulq_n_f32(g20, 0.5));
    float32x4_t dst11 = vaddq_f32(vmulq_n_f32(g01, 0.5), vmulq_n_f32(g11, 0.5));
    dst11 = vaddq_f32(dst11, vmulq_n_f32(g21, 0.5));
    float32x4_t dst12 = vaddq_f32(vmulq_n_f32(g02, 0.5), vmulq_n_f32(g12, 0.5));
    dst12 = vaddq_f32(dst12, vmulq_n_f32(g22, 0.5));

    float32x4_t dst20 = vsubq_f32(vmulq_n_f32(g00, 0.5), vmulq_n_f32(g10, 0.5));
    dst20 = vaddq_f32(dst20, vmulq_n_f32(g20, 0.5));
    float32x4_t dst21 = vsubq_f32(vmulq_n_f32(g01, 0.5), vmulq_n_f32(g11, 0.5));
    dst21 = vaddq_f32(dst21, vmulq_n_f32(g21, 0.5));
    float32x4_t dst22 = vsubq_f32(vmulq_n_f32(g02, 0.5), vmulq_n_f32(g12, 0.5));
    dst22 = vaddq_f32(dst22, vmulq_n_f32(g22, 0.5));

    float32x4_t dst30 = g20;
    float32x4_t dst31 = g21;
    float32x4_t dst32 = g22;

    float32x4_t m00 = dst00;
    float32x4_t m01 = vaddq_f32(vmulq_n_f32(dst00, 0.5), vmulq_n_f32(dst01, 0.5));
    m01 = vaddq_f32(m01, vmulq_n_f32(dst02, 0.5));
    float32x4_t m02 = vsubq_f32(vmulq_n_f32(dst00, 0.5), vmulq_n_f32(dst01, 0.5));
    m02 = vaddq_f32(m02, vmulq_n_f32(dst02, 0.5));
    float32x4_t m03 = dst02;

    float32x4_t m10 = dst10;
    float32x4_t m11 = vaddq_f32(vmulq_n_f32(dst10, 0.5), vmulq_n_f32(dst11, 0.5));
    m11 = vaddq_f32(m11, vmulq_n_f32(dst12, 0.5));
    float32x4_t m12 = vsubq_f32(vmulq_n_f32(dst10, 0.5), vmulq_n_f32(dst11, 0.5));
    m12 = vaddq_f32(m12, vmulq_n_f32(dst12, 0.5));
    float32x4_t m13 = dst12;

    float32x4_t m20 = dst20;
    float32x4_t m21 = vaddq_f32(vmulq_n_f32(dst20, 0.5), vmulq_n_f32(dst21, 0.5));
    m21 = vaddq_f32(m21, vmulq_n_f32(dst22, 0.5));
    float32x4_t m22 = vsubq_f32(vmulq_n_f32(dst20, 0.5), vmulq_n_f32(dst21, 0.5));
    m22 = vaddq_f32(m22, vmulq_n_f32(dst22, 0.5));
    float32x4_t m23 = dst22;

    float32x4_t m30 = dst30;
    float32x4_t m31 = vaddq_f32(vmulq_n_f32(dst30, 0.5), vmulq_n_f32(dst31, 0.5));
    m31 = vaddq_f32(m31, vmulq_n_f32(dst32, 0.5));
    float32x4_t m32 = vsubq_f32(vmulq_n_f32(dst30, 0.5), vmulq_n_f32(dst31, 0.5));
    m32 = vaddq_f32(m32, vmulq_n_f32(dst32, 0.5));
    float32x4_t m33 = dst32;

    vst1q_f32(dst, m00);
    vst1q_f32(dst + 4, m01);
    vst1q_f32(dst + 8, m02);
    vst1q_f32(dst + 12, m03);
    vst1q_f32(dst + 16, m10);
    vst1q_f32(dst + 20, m11);
    vst1q_f32(dst + 24, m12);
    vst1q_f32(dst + 28, m13);
    vst1q_f32(dst + 32, m20);
    vst1q_f32(dst + 36, m21);
    vst1q_f32(dst + 40, m22);
    vst1q_f32(dst + 44, m23);
    vst1q_f32(dst + 48, m30);
    vst1q_f32(dst + 52, m31);
    vst1q_f32(dst + 56, m32);
    vst1q_f32(dst + 60, m33);
#else
    for (int j = 0; j < C4NUM; j++) {
      float *local_ptr = src + j;
      float dst00 = local_ptr[0];
      float dst01 = (local_ptr + 4)[0];
      float dst02 = (local_ptr + 8)[0];

      const float dst10 = 0.5f * local_ptr[0] + 0.5f * (local_ptr + 12)[0] + 0.5f * (local_ptr + 24)[0];
      const float dst11 = 0.5f * (local_ptr + 4)[0] + 0.5f * (local_ptr + 16)[0] + 0.5f * (local_ptr + 28)[0];
      const float dst12 = 0.5f * (local_ptr + 8)[0] + 0.5f * (local_ptr + 20)[0] + 0.5f * (local_ptr + 32)[0];

      const float dst20 = 0.5f * local_ptr[0] - 0.5f * (local_ptr + 12)[0] + 0.5f * (local_ptr + 24)[0];
      const float dst21 = 0.5f * (local_ptr + 4)[0] - 0.5f * (local_ptr + 16)[0] + 0.5f * (local_ptr + 28)[0];
      const float dst22 = 0.5f * (local_ptr + 8)[0] - 0.5f * (local_ptr + 20)[0] + 0.5f * (local_ptr + 32)[0];

      float dst30 = (local_ptr + 24)[0];
      float dst31 = (local_ptr + 28)[0];
      float dst32 = (local_ptr + 32)[0];

      float m00 = dst00;
      const float m01 = 0.5f * dst00 + 0.5f * dst01 + 0.5f * dst02;
      const float m02 = 0.5f * dst00 - 0.5f * dst01 + 0.5f * dst02;
      float m03 = dst02;

      float m10 = dst10;
      const float m11 = 0.5f * dst10 + 0.5f * dst11 + 0.5f * dst12;
      const float m12 = 0.5f * dst10 - 0.5f * dst11 + 0.5f * dst12;
      float m13 = dst12;

      float m20 = dst20;
      const float m21 = 0.5f * dst20 + 0.5f * dst21 + 0.5f * dst22;
      const float m22 = 0.5f * dst20 - 0.5f * dst21 + 0.5f * dst22;
      float m23 = dst22;

      float m30 = dst30;
      const float m31 = 0.5f * dst30 + 0.5f * dst31 + 0.5f * dst32;
      const float m32 = 0.5f * dst30 - 0.5f * dst31 + 0.5f * dst32;
      float m33 = dst32;

      *(dst + j) = m00;
      *(dst + j + 4) = m01;
      *(dst + j + 8) = m02;
      *(dst + j + 12) = m03;

      *(dst + j + 16) = m10;
      *(dst + j + 20) = m11;
      *(dst + j + 24) = m12;
      *(dst + j + 28) = m13;

      *(dst + j + 32) = m20;
      *(dst + j + 36) = m21;
      *(dst + j + 40) = m22;
      *(dst + j + 44) = m23;

      *(dst + j + 48) = m30;
      *(dst + j + 52) = m31;
      *(dst + j + 56) = m32;
      *(dst + j + 60) = m33;
    }
#endif
  }
}

void ConvDw3x3Fp32InputTrans(const float *input_data, float *trans_input, float *block_buffer, int out_h_block,
                             int out_w_block, const ConvParameter *conv_param) {
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  const int input_unit = 4;
  memset(trans_input, 0, out_h_block * out_h_block * 16 * C4NUM * sizeof(float));

  for (int oh = 0; oh < out_h_block; oh++) {
    int ih = oh * 2 - conv_param->pad_h_;
    int real_h_start = ih > 0 ? 0 : -ih;
    int real_h_end = (ih + input_unit) < conv_param->input_h_ ? input_unit : (conv_param->input_h_ - ih);
    for (int ow = 0; ow < out_w_block; ow++) {
      int iw = ow * 2 - conv_param->pad_w_;
      int real_w_start = iw > 0 ? 0 : -iw;
      int real_w_end = (iw + input_unit) < conv_param->input_w_ ? input_unit : (conv_param->input_w_ - iw);

      memset(block_buffer, 0, 16 * C4NUM * sizeof(float));
      int src_plane_offset = ic4 * C4NUM * (ih * conv_param->input_w_ + iw);
      for (int h = real_h_start; h < real_h_end; h++) {
        int src_h_offset = src_plane_offset + (h * conv_param->input_w_) * ic4 * C4NUM;
        int dst_h_offset = (h * input_unit) * C4NUM;
        for (int w = real_w_start; w < real_w_end; w++) {
          int src_w_offset = src_h_offset + w * ic4 * C4NUM;
          int dst_w_offset = dst_h_offset + w * C4NUM;
          float *src_addr = (float *)(input_data) + src_w_offset;
          float *dst_addr = block_buffer + dst_w_offset;
#ifdef ENABLE_NEON
          vst1q_f32(dst_addr, vld1q_f32(src_addr));
#else
          for (int k = 0; k < C4NUM; k++) {
            (dst_addr + k)[0] = (src_addr + k)[0];
          }
#endif
        }
      }
      int trans_offset = (oh * out_w_block + ow) * 16 * C4NUM;
      Conv3x3Fp32InputUnit(block_buffer, trans_input + trans_offset, C4NUM);
    }
  }
}

// todo yangruoqi: implement assembly
void ConvDw3x3Fp32Winograd(float *trans_buffer, const float *weight, int out_h_block, int out_w_block) {
  const int unit = 4;
  for (int oh = 0; oh < out_h_block; oh++) {
    float *buf_oh = trans_buffer + oh * out_w_block * 16 * C4NUM;
    for (int ow = 0; ow < out_w_block; ow++) {
      float *buf_ow = buf_oh + ow * 16 * C4NUM;
      for (int kh = 0; kh < unit; kh++) {
        float *buf_kh = buf_ow + kh * unit * C4NUM;
        const float *weight_kh = weight + kh * unit * C4NUM;
        for (int kw = 0; kw < unit; kw++) {
          float *buf_kw = buf_kh + kw * C4NUM;
          const float *weight_kw = weight_kh + kw * C4NUM;
          for (int c = 0; c < C4NUM; c++) {
            buf_kw[c] = buf_kw[c] * weight_kw[c];
          }
        }
      }
    }
  }
}

void ConvDw3x3Fp32OutputUnit(float *src_buf, float *dst_output, const float *bias, int channel, int output_w,
                             bool h_in_range, bool w_in_range, bool is_relu, bool is_relu6) {
#ifdef ENABLE_ARM
  float32x4_t bias_ptr = vld1q_f32(bias);

  float32x4_t s00 = vld1q_f32(src_buf);
  float32x4_t s01 = vld1q_f32(src_buf + 4);
  float32x4_t s02 = vld1q_f32(src_buf + 8);
  float32x4_t s03 = vld1q_f32(src_buf + 12);

  float32x4_t s10 = vld1q_f32(src_buf + 16);
  float32x4_t s11 = vld1q_f32(src_buf + 20);
  float32x4_t s12 = vld1q_f32(src_buf + 24);
  float32x4_t s13 = vld1q_f32(src_buf + 28);

  float32x4_t s20 = vld1q_f32(src_buf + 32);
  float32x4_t s21 = vld1q_f32(src_buf + 36);
  float32x4_t s22 = vld1q_f32(src_buf + 40);
  float32x4_t s23 = vld1q_f32(src_buf + 44);

  float32x4_t s30 = vld1q_f32(src_buf + 48);
  float32x4_t s31 = vld1q_f32(src_buf + 52);
  float32x4_t s32 = vld1q_f32(src_buf + 56);
  float32x4_t s33 = vld1q_f32(src_buf + 60);

  float32x4_t t00 = vaddq_f32(vaddq_f32(s00, s10), s20);
  float32x4_t t01 = vaddq_f32(vaddq_f32(s01, s11), s21);
  float32x4_t t02 = vaddq_f32(vaddq_f32(s02, s12), s22);
  float32x4_t t03 = vaddq_f32(vaddq_f32(s03, s13), s23);

  float32x4_t t10 = vsubq_f32(vsubq_f32(s10, s20), s30);
  float32x4_t t11 = vsubq_f32(vsubq_f32(s11, s21), s31);
  float32x4_t t12 = vsubq_f32(vsubq_f32(s12, s22), s32);
  float32x4_t t13 = vsubq_f32(vsubq_f32(s13, s23), s33);

  float32x4_t d00 = vaddq_f32(vaddq_f32(vaddq_f32(t00, t01), t02), bias_ptr);
  float32x4_t d01 = vaddq_f32(vsubq_f32(vsubq_f32(t01, t02), t03), bias_ptr);
  float32x4_t d10 = vaddq_f32(vaddq_f32(vaddq_f32(t10, t11), t12), bias_ptr);
  float32x4_t d11 = vaddq_f32(vsubq_f32(vsubq_f32(t11, t12), t13), bias_ptr);

  float32x4_t zeros = {0, 0, 0, 0};
  float32x4_t bounds = {6, 6, 6, 6};
  if (is_relu) {
    d00 = vmaxq_f32(d00, zeros);
    d01 = vmaxq_f32(d01, zeros);
    d10 = vmaxq_f32(d10, zeros);
    d11 = vmaxq_f32(d11, zeros);
  }
  if (is_relu6) {
    d00 = vminq_f32(vmaxq_f32(d00, zeros), bounds);
    d01 = vminq_f32(vmaxq_f32(d01, zeros), bounds);
    d10 = vminq_f32(vmaxq_f32(d10, zeros), bounds);
    d11 = vminq_f32(vmaxq_f32(d11, zeros), bounds);
  }

  vst1q_f32(dst_output, d00);
  if (w_in_range) {
    vst1q_f32(dst_output + channel, d01);
  }
  if (h_in_range) {
    vst1q_f32(dst_output + output_w * channel, d10);
    if (w_in_range) {
      vst1q_f32(dst_output + output_w * channel + channel, d11);
    }
  }
#else
  for (int i = 0; i < C4NUM; i++) {
    const float *local_ptr = src_buf + i;
    const float *bias_ptr = bias + i;

    float s00 = local_ptr[0];
    float s01 = (local_ptr + 4)[0];
    float s02 = (local_ptr + 8)[0];
    float s03 = (local_ptr + 12)[0];

    float s10 = (local_ptr + 16)[0];
    float s11 = (local_ptr + 20)[0];
    float s12 = (local_ptr + 24)[0];
    float s13 = (local_ptr + 28)[0];

    float s20 = (local_ptr + 32)[0];
    float s21 = (local_ptr + 36)[0];
    float s22 = (local_ptr + 40)[0];
    float s23 = (local_ptr + 44)[0];

    float s30 = (local_ptr + 48)[0];
    float s31 = (local_ptr + 52)[0];
    float s32 = (local_ptr + 56)[0];
    float s33 = (local_ptr + 60)[0];

    float t00 = s00 + s10 + s20;
    float t01 = s01 + s11 + s21;
    float t02 = s02 + s12 + s22;
    float t03 = s03 + s13 + s23;

    float t10 = s10 - s20 - s30;
    float t11 = s11 - s21 - s31;
    float t12 = s12 - s22 - s32;
    float t13 = s13 - s23 - s33;

    float d00 = t00 + t01 + t02 + bias_ptr[0];
    float d01 = t01 - t02 - t03 + bias_ptr[0];
    float d10 = t10 + t11 + t12 + bias_ptr[0];
    float d11 = t11 - t12 - t13 + bias_ptr[0];

    if (is_relu) {
      d00 = MSMAX(d00, 0);
      d01 = MSMAX(d01, 0);
      d10 = MSMAX(d10, 0);
      d11 = MSMAX(d11, 0);
    }
    if (is_relu6) {
      d00 = MSMIN(MSMAX(d00, 0), 6);
      d01 = MSMIN(MSMAX(d01, 0), 6);
      d10 = MSMIN(MSMAX(d10, 0), 6);
      d11 = MSMIN(MSMAX(d11, 0), 6);
    }

    (dst_output + i)[0] = d00;
    if (w_in_range) {
      (dst_output + i + channel)[0] = d01;
    }
    if (h_in_range) {
      (dst_output + i + output_w * channel)[0] = d10;
      if (w_in_range) {
        (dst_output + i + output_w * channel + channel)[0] = d11;
      }
    }
  }
#endif
}

void ConvDw3x3Fp32OutputTrans(float *trans_buffer, float *output_data, const float *bias, int out_h_block,
                              int out_w_block, const ConvParameter *conv_param) {
  int oc4 = UP_DIV(conv_param->output_channel_, C4NUM);
  bool h_in_range = true;
  for (int oh = 0; oh < out_h_block; oh++) {
    const int real_oh = 2 * oh;
    if ((oh + 1) * 2 > conv_param->output_h_) {
      h_in_range = false;
    }
    bool w_in_range = true;
    float *buf_oh = trans_buffer + oh * out_w_block * 16 * C4NUM;
    float *output_oh = output_data + real_oh * conv_param->output_w_ * oc4 * C4NUM;

    for (int ow = 0; ow < out_w_block; ow++) {
      const int real_ow = 2 * ow;
      if ((ow + 1) * 2 > conv_param->output_w_) {
        w_in_range = false;
      }
      float *buf_ow = buf_oh + ow * 16 * C4NUM;
      float *output_ow = output_oh + real_ow * oc4 * C4NUM;

      ConvDw3x3Fp32OutputUnit(buf_ow, output_ow, bias, oc4 * C4NUM, conv_param->output_w_, h_in_range, w_in_range,
                              conv_param->is_relu_, conv_param->is_relu6_);
    }
  }
}

void ConvDw3x3Fp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                   float *trans_buffer, float *block_buffer, const ConvParameter *conv_param, int task_id) {
  int thread_count = conv_param->thread_num_;
  int output_channel = conv_param->output_channel_;
  int oc4 = UP_DIV(output_channel, C4NUM);
  int out_h_block = UP_DIV(conv_param->output_h_, 2);
  int out_w_block = UP_DIV(conv_param->output_w_, 2);

  int input_batch = conv_param->input_batch_;
  for (int batch = 0; batch < input_batch; batch++) {
    const float *input = input_data + batch * conv_param->input_h_ * conv_param->input_w_ *
                                        UP_DIV(conv_param->input_channel_, C4NUM) * C4NUM;
    float *output = output_data + batch * conv_param->output_h_ * conv_param->output_w_ *
                                    UP_DIV(conv_param->output_channel_, C4NUM) * C4NUM;
    for (int oc = task_id; oc < oc4; oc += thread_count) {
      const float *weight = weight_data + oc * 16 * C4NUM;
      const float *bias = bias_data + oc * C4NUM;

      ConvDw3x3Fp32InputTrans(input + oc * C4NUM, trans_buffer, block_buffer, out_h_block, out_w_block, conv_param);

      ConvDw3x3Fp32Winograd(trans_buffer, weight, out_h_block, out_w_block);

      ConvDw3x3Fp32OutputTrans(trans_buffer, output + oc * C4NUM, bias, out_h_block, out_w_block, conv_param);
    }
  }
}
/*conv depthwise 3x3 fp32 end*/

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
