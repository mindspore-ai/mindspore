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

#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/common_func.h"
#include "nnacl/fp32/common_func_fp32.h"

#if !defined(ENABLE_ARM) && !defined(ENABLE_SSE)
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

  while (left * conv_param->stride_w_ < conv_param->pad_l_) {
    left++;
  }
  while ((right - 1) * conv_param->stride_w_ - conv_param->pad_l_ + conv_param->kernel_w_ * conv_param->dilation_w_ >
           conv_param->input_w_ &&
         right > left) {
    right--;
  }
  while (top * conv_param->stride_h_ < conv_param->pad_u_) {
    top++;
  }
  while ((bottom - 1) * conv_param->stride_h_ - conv_param->pad_u_ + conv_param->kernel_h_ * conv_param->dilation_h_ >
           conv_param->input_h_ &&
         bottom > top) {
    bottom--;
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
void ConvDwBorderPixel(float *dst, const float *src, const float *weight, const float *bias, int height, int width,
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

void ConvDwBorder(float *dst, const float *src, const float *weight, const float *bias, int top, int bottom, int left,
                  int right, const ConvParameter *conv_param, const SlidingWindowParam *sliding) {
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
#ifdef ENABLE_AVX
      ConvDwFp32BorderParam *param = (ConvDwFp32BorderParam *)malloc(sizeof(ConvDwFp32BorderParam));
      param->dst = dst_kernel;
      param->src = src_kernel;
      param->weight = weight_kernel;
      param->bias = bias;
      param->height = end_kh - start_kh;
      param->width = end_kw - start_kw;
      param->in_kh_step = sliding->in_kh_step_ * sizeof(float);
      param->in_kw_step = sliding->in_kw_step_ * sizeof(float);
      param->kernel_w = conv_param->kernel_w_ * C4NUM * sizeof(float);
      param->relu = relu;
      param->relu6 = relu6;
      ConvDwFp32Border(param);
      free(param);
#elif defined(ENABLE_ARM) || defined(ENABLE_SSE)
      ConvDwFp32Border(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                       sliding->in_kh_step_ * sizeof(float), sliding->in_kw_step_ * sizeof(float),
                       conv_param->kernel_w_ * C4NUM * sizeof(float), relu, relu6);
#else
      ConvDwBorderPixel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw,
                        sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_ * C4NUM, relu, relu6);
#endif
      dst_kernel += sliding->block_channel_;
    }  // width loop
    dst_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void ConvDwCenter(float *dst, const float *src, const float *weight, const float *bias, int height, int width,
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
void ConvDwSWFp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
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
      ConvDwBorder(dst_data, src_data, weight, bias, 0, sliding->top_, 0, conv_param->output_w_, conv_param, sliding);
      ConvDwBorder(dst_data, src_data, weight, bias, sliding->bottom_, conv_param->output_h_, 0, conv_param->output_w_,
                   conv_param, sliding);
      ConvDwBorder(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, 0, sliding->left_, conv_param,
                   sliding);
      ConvDwBorder(dst_data, src_data, weight, bias, sliding->top_, sliding->bottom_, sliding->right_,
                   conv_param->output_w_, conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int in_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int in_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        const float *in_t = src_data + in_h_start * sliding->in_h_step_ + in_w_start * sliding->block_channel_;
        float *out_t = dst_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;
#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
        ConvDwFp32Center(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                         conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(float),
                         sliding->block_channel_ * sizeof(float), sliding->in_sh_step_ * sizeof(float),
                         sliding->in_sw_step_ * sizeof(float), sliding->in_kh_step_ * sizeof(float),
                         sliding->in_kw_step_ * sizeof(float), relu, relu6);
#else
        ConvDwCenter(out_t, in_t, weight, bias, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
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

/*conv depthwise 3x3 fp32 begin*/
bool CheckConvDwUse3X3(const ConvParameter *conv_param) {
  bool use_3x3 =
    conv_param->kernel_h_ == 3 && conv_param->kernel_w_ == 3 &&
    (conv_param->stride_h_ == 1 || conv_param->stride_h_ == 2) &&
    (conv_param->stride_w_ == 1 || conv_param->stride_w_ == 2) && conv_param->stride_h_ == conv_param->stride_w_ &&
    (conv_param->pad_u_ == 0 || conv_param->pad_u_ == 1) && (conv_param->pad_l_ == 0 || conv_param->pad_l_ == 1) &&
    conv_param->pad_u_ == conv_param->pad_l_ && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1;
  if (!use_3x3 || conv_param->input_h_ == 1 || conv_param->input_w_ == 1) {
    return false;
  }
  const int in_h = (conv_param->output_h_ - 1) * conv_param->stride_h_ + conv_param->kernel_h_;
  const int in_w = (conv_param->output_w_ - 1) * conv_param->stride_w_ + conv_param->kernel_w_;
  return in_h == (conv_param->input_h_ + 2 * conv_param->pad_u_) &&
         in_w == (conv_param->input_w_ + 2 * conv_param->pad_l_);
}

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
bool CheckConvDw1DWinograd(const ConvParameter *conv_param, int thread_num) {
  return conv_param->kernel_h_ == 3 && conv_param->kernel_w_ == 3 && conv_param->stride_w_ == 1 &&
         conv_param->stride_h_ == 1 && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
         conv_param->pad_u_ == 1 && conv_param->pad_d_ == 1 && conv_param->pad_l_ == 1 && conv_param->pad_r_ == 1 &&
         conv_param->input_channel_ == conv_param->output_channel_ && conv_param->output_w_ >= 4 &&
         conv_param->output_h_ >= thread_num * 4;  // better had more than 4 rows for each thread
}

void ConvDw3x3RowLeft(const float *src, float *line, int lw, int channel) {
  MS_FLOAT32X4 v0, v1, v2, v3;
  v0 = MS_MOVQ_F32(0.0f);
  int ic = 0;
  for (; ic < channel - 3; ic += 4) {
    v1 = MS_LDQ_F32(src + ic);
    v2 = MS_LDQ_F32(src + channel + ic);
    v3 = MS_LDQ_F32(src + 2 * channel + ic);
    MS_FLOAT32X4 b0 = MS_SUBQ_F32(v0, v2);
    MS_FLOAT32X4 b1 = MS_ADDQ_F32(v1, v2);
    MS_FLOAT32X4 b2 = MS_SUBQ_F32(v2, v1);
    MS_FLOAT32X4 b3 = MS_SUBQ_F32(v3, v1);
    MS_STQ_F32(line + lw * ic, b0);
    MS_STQ_F32(line + lw * ic + 4, b1);
    MS_STQ_F32(line + lw * ic + 8, b2);
    MS_STQ_F32(line + lw * ic + 12, b3);
  }
  if (ic < channel) {
    float *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 4, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 12, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float d1 = src[i + ic];
      float d2 = src[i + ic + channel];
      float d3 = src[i + ic + 2 * channel];
      remain_line[i] = 0.0f - d2;
      remain_line[i + 4] = d1 + d2;
      remain_line[i + 8] = d2 - d1;
      remain_line[i + 12] = d3 - d1;
    }
  }
}

void ConvDw3x3RowMiddle(const float *src, float *line, int lw, int channel) {
  MS_FLOAT32X4 v0, v1, v2, v3;
  int ic = 0;
  for (; ic < channel - 3; ic += 4) {
    v0 = MS_LDQ_F32(src + ic);
    v1 = MS_LDQ_F32(src + channel + ic);
    v2 = MS_LDQ_F32(src + 2 * channel + ic);
    v3 = MS_LDQ_F32(src + 3 * channel + ic);
    MS_FLOAT32X4 b0 = MS_SUBQ_F32(v0, v2);
    MS_FLOAT32X4 b1 = MS_ADDQ_F32(v1, v2);
    MS_FLOAT32X4 b2 = MS_SUBQ_F32(v2, v1);
    MS_FLOAT32X4 b3 = MS_SUBQ_F32(v3, v1);
    MS_STQ_F32(line + lw * ic, b0);
    MS_STQ_F32(line + lw * ic + 4, b1);
    MS_STQ_F32(line + lw * ic + 8, b2);
    MS_STQ_F32(line + lw * ic + 12, b3);
  }
  if (ic < channel) {
    float *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 4, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 12, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float d0 = src[i + ic];
      float d1 = src[i + ic + channel];
      float d2 = src[i + ic + 2 * channel];
      float d3 = src[i + ic + 3 * channel];
      remain_line[i] = d0 - d2;
      remain_line[i + 4] = d1 + d2;
      remain_line[i + 8] = d2 - d1;
      remain_line[i + 12] = d3 - d1;
    }
  }
}

void ConvDw3x3RowRight(const float *src, float *line, int lw, int channel) {
  MS_FLOAT32X4 v0, v1, v2, v3;
  int ic = 0;
  v3 = MS_MOVQ_F32(0.0f);
  for (; ic < channel - 3; ic += 4) {
    v0 = MS_LDQ_F32(src + ic);
    v1 = MS_LDQ_F32(src + channel + ic);
    v2 = MS_LDQ_F32(src + 2 * channel + ic);
    MS_FLOAT32X4 b0 = MS_SUBQ_F32(v0, v2);
    MS_FLOAT32X4 b1 = MS_ADDQ_F32(v1, v2);
    MS_FLOAT32X4 b2 = MS_SUBQ_F32(v2, v1);
    MS_FLOAT32X4 b3 = MS_SUBQ_F32(v3, v1);
    MS_STQ_F32(line + lw * ic, b0);
    MS_STQ_F32(line + lw * ic + 4, b1);
    MS_STQ_F32(line + lw * ic + 8, b2);
    MS_STQ_F32(line + lw * ic + 12, b3);
  }
  if (ic < channel) {
    float *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 4, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 12, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float d0 = src[i + ic];
      float d1 = src[i + ic + channel];
      float d2 = src[i + ic + 2 * channel];
      remain_line[i] = d0 - d2;
      remain_line[i + 4] = d1 + d2;
      remain_line[i + 8] = d2 - d1;
      remain_line[i + 12] = 0.0f - d1;
    }
  }
}

void ConvDw3x3RowSingle(const float *src, float *line, int lw, int channel) {
  MS_FLOAT32X4 v0, v1, v2;
  int ic = 0;
  v2 = MS_MOVQ_F32(0.0f);
  for (; ic < channel - 3; ic += 4) {
    v0 = MS_LDQ_F32(src + ic);
    v1 = MS_LDQ_F32(src + channel + ic);
    MS_FLOAT32X4 b2 = MS_SUBQ_F32(v2, v1);
    MS_STQ_F32(line + lw * ic, v0);
    MS_STQ_F32(line + lw * ic + 4, v1);
    MS_STQ_F32(line + lw * ic + 8, b2);
    memset(line + lw * ic + 12, 0, 16);
  }
  if (ic < channel) {
    float *remain_line = line + ic * lw;
    memset(remain_line, 0, 16);
    memset(remain_line + 4, 0, 16);
    memset(remain_line + 8, 0, 16);
    memset(remain_line + 12, 0, 16);
    for (int i = 0; i < channel - ic; i++) {
      float d0 = src[i + ic];
      float d1 = src[i + ic + channel];
      remain_line[i] = d0;
      remain_line[i + 4] = d1;
      remain_line[i + 8] = 0.0f - d1;
    }
  }
}

void ConvDw3x3InitTop(const float *src, float **lines, int width, int channel) {
  float *line0 = lines[0];
  float *line1 = lines[1];
  float *line2 = lines[2];
  int c4 = UP_ROUND(channel, C4NUM);
  int lw = UP_DIV(width, C2NUM) * C4NUM;
  memset(line0, 0, c4 * lw * sizeof(float));
  ConvDw3x3RowLeft(src, line1, lw, channel);
  ConvDw3x3RowLeft(src + width * channel, line2, lw, channel);
  int ow = 2;
  for (; ow < width - 2; ow += 2) {
    ConvDw3x3RowMiddle(src + (ow - 1) * channel, line1 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowMiddle(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 4, lw, channel);
  }
  int remain = width - ow;
  if (remain == 2) {
    ConvDw3x3RowRight(src + (ow - 1) * channel, line1 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowRight(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 4, lw, channel);
  } else if (remain == 1) {
    ConvDw3x3RowSingle(src + (ow - 1) * channel, line1 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowSingle(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 4, lw, channel);
  }
}

void ConvDw3x3InitRow(const float *src, float **lines, int width, int channel) {
  float *line0 = lines[0];
  float *line1 = lines[1];
  float *line2 = lines[2];
  int lw = UP_DIV(width, C2NUM) * C4NUM;
  ConvDw3x3RowLeft(src - width * channel, line0, lw, channel);
  ConvDw3x3RowLeft(src, line1, lw, channel);
  ConvDw3x3RowLeft(src + width * channel, line2, lw, channel);
  int ow = 2;
  for (; ow < width - 2; ow += 2) {
    ConvDw3x3RowMiddle(src - width * channel + (ow - 1) * channel, line0 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowMiddle(src + (ow - 1) * channel, line1 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowMiddle(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 4, lw, channel);
  }
  int remain = width - ow;
  if (remain == 2) {
    ConvDw3x3RowRight(src - width * channel + (ow - 1) * channel, line0 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowRight(src + (ow - 1) * channel, line1 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowRight(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 4, lw, channel);
  } else if (remain == 1) {
    ConvDw3x3RowSingle(src - width * channel + (ow - 1) * channel, line0 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowSingle(src + (ow - 1) * channel, line1 + 2 * ow * 4, lw, channel);
    ConvDw3x3RowSingle(src + width * channel + (ow - 1) * channel, line2 + 2 * ow * 4, lw, channel);
  }
}

void ConvDw3x3Row(const float *src, float **lines, int width, int channel) {
  float *tmp = lines[0];
  lines[0] = lines[1];
  lines[1] = lines[2];
  lines[2] = tmp;
  int c4 = UP_ROUND(channel, C4NUM);
  int lw = UP_DIV(width, C2NUM) * C4NUM;
  memset(tmp, 0, c4 * lw * sizeof(float));
  ConvDw3x3RowLeft(src, tmp, lw, channel);
  int ow = 2;
  for (; ow < width - 2; ow += 2) {
    ConvDw3x3RowMiddle(src + (ow - 1) * channel, tmp + 2 * ow * 4, lw, channel);
  }
  int remain = width - ow;
  if (remain == 2) {
    ConvDw3x3RowRight(src + (ow - 1) * channel, tmp + 2 * ow * 4, lw, channel);
  } else if (remain == 1) {
    ConvDw3x3RowSingle(src + (ow - 1) * channel, tmp + 2 * ow * 4, lw, channel);
  }
}

void ConvDw3x3Bottom(float **lines, int width, int channel) {
  float *tmp = lines[0];
  lines[0] = lines[1];
  lines[1] = lines[2];
  lines[2] = tmp;
  int c4 = UP_ROUND(channel, C4NUM);
  memset(tmp, 0, UP_DIV(width, C2NUM) * c4 * C4NUM * sizeof(float));
}

void ConvDw3x3Line(float *dst, float **lines, const float *weight, const float *bias_data, int width, int ori_channel,
                   bool relu, bool relu6) {
  int channel = ori_channel;
  float *line0 = lines[0];
  float *line1 = lines[1];
  float *line2 = lines[2];
  for (; channel > 0; channel -= 4) {
    MS_FLOAT32X4 bias = MS_LDQ_F32(bias_data);
    bias_data += 4;
    MS_FLOAT32X4 g00 = MS_LDQ_F32(weight);
    MS_FLOAT32X4 g01 = MS_LDQ_F32(weight + 4);
    MS_FLOAT32X4 g02 = MS_LDQ_F32(weight + 8);
    MS_FLOAT32X4 g03 = MS_LDQ_F32(weight + 12);
    MS_FLOAT32X4 g10 = MS_LDQ_F32(weight + 16);
    MS_FLOAT32X4 g11 = MS_LDQ_F32(weight + 20);
    MS_FLOAT32X4 g12 = MS_LDQ_F32(weight + 24);
    MS_FLOAT32X4 g13 = MS_LDQ_F32(weight + 28);
    MS_FLOAT32X4 g20 = MS_LDQ_F32(weight + 32);
    MS_FLOAT32X4 g21 = MS_LDQ_F32(weight + 36);
    MS_FLOAT32X4 g22 = MS_LDQ_F32(weight + 40);
    MS_FLOAT32X4 g23 = MS_LDQ_F32(weight + 44);
    weight += 48;
    float *cur_dst = dst;
    int ow = 0;
    for (; ow < width - 1; ow += 2) {
      MS_FLOAT32X4 acc0 = MS_MULQ_F32(MS_LDQ_F32(line0), g00);
      MS_FLOAT32X4 acc1 = MS_MULQ_F32(MS_LDQ_F32(line0 + 4), g01);
      MS_FLOAT32X4 acc2 = MS_MULQ_F32(MS_LDQ_F32(line0 + 8), g02);
      MS_FLOAT32X4 acc3 = MS_MULQ_F32(MS_LDQ_F32(line0 + 12), g03);
      line0 += 16;
      acc0 = MS_MLAQ_F32(acc0, MS_LDQ_F32(line1), g10);
      acc1 = MS_MLAQ_F32(acc1, MS_LDQ_F32(line1 + 4), g11);
      acc2 = MS_MLAQ_F32(acc2, MS_LDQ_F32(line1 + 8), g12);
      acc3 = MS_MLAQ_F32(acc3, MS_LDQ_F32(line1 + 12), g13);
      line1 += 16;
      acc0 = MS_MLAQ_F32(acc0, MS_LDQ_F32(line2), g20);
      acc1 = MS_MLAQ_F32(acc1, MS_LDQ_F32(line2 + 4), g21);
      acc2 = MS_MLAQ_F32(acc2, MS_LDQ_F32(line2 + 8), g22);
      acc3 = MS_MLAQ_F32(acc3, MS_LDQ_F32(line2 + 12), g23);
      line2 += 16;
      MS_FLOAT32X4 res0 = MS_ADDQ_F32(acc0, MS_ADDQ_F32(acc2, acc1));
      MS_FLOAT32X4 res1 = MS_ADDQ_F32(acc1, MS_SUBQ_F32(acc3, acc2));
      res0 = MS_ADDQ_F32(res0, bias);
      res1 = MS_ADDQ_F32(res1, bias);
      if (relu || relu6) {
        res0 = MS_MAXQ_F32(res0, MS_MOVQ_F32(0.0f));
        res1 = MS_MAXQ_F32(res1, MS_MOVQ_F32(0.0f));
      }
      if (relu6) {
        res0 = MS_MINQ_F32(res0, MS_MOVQ_F32(6.0f));
        res1 = MS_MINQ_F32(res1, MS_MOVQ_F32(6.0f));
      }
      if (channel >= 4) {
        MS_STQ_F32(cur_dst, res0);
        MS_STQ_F32(cur_dst + ori_channel, res1);
      } else {
        for (int i = 0; i < channel; i++) {
          cur_dst[i] = res0[i];
          cur_dst[ori_channel + i] = res1[i];
        }
      }
      cur_dst += 2 * ori_channel;
    }
    if (ow < width) {
      MS_FLOAT32X4 acc0 = MS_MULQ_F32(MS_LDQ_F32(line0), g00);
      MS_FLOAT32X4 acc1 = MS_MULQ_F32(MS_LDQ_F32(line0 + 4), g01);
      MS_FLOAT32X4 acc2 = MS_MULQ_F32(MS_LDQ_F32(line0 + 8), g02);
      line0 += 16;
      acc0 = MS_MLAQ_F32(acc0, MS_LDQ_F32(line1), g10);
      acc1 = MS_MLAQ_F32(acc1, MS_LDQ_F32(line1 + 4), g11);
      acc2 = MS_MLAQ_F32(acc2, MS_LDQ_F32(line1 + 8), g12);
      line1 += 16;
      acc0 = MS_MLAQ_F32(acc0, MS_LDQ_F32(line2), g20);
      acc1 = MS_MLAQ_F32(acc1, MS_LDQ_F32(line2 + 4), g21);
      acc2 = MS_MLAQ_F32(acc2, MS_LDQ_F32(line2 + 8), g22);
      line2 += 16;
      MS_FLOAT32X4 res0 = MS_ADDQ_F32(acc0, MS_ADDQ_F32(acc2, acc1));
      res0 = MS_ADDQ_F32(res0, bias);
      if (relu || relu6) {
        res0 = MS_MAXQ_F32(res0, MS_MOVQ_F32(0.0f));
      }
      if (relu6) {
        res0 = MS_MINQ_F32(res0, MS_MOVQ_F32(6.0f));
      }
      if (channel >= 4) {
        MS_STQ_F32(cur_dst, res0);
      } else {
        for (int i = 0; i < channel; i++) {
          cur_dst[i] = res0[i];
        }
      }
    }
    dst += 4;
  }
}

void ConvDw3x3(float *output_data, float *buffer, const float *input_data, const float *weight_data,
               const float *bias_data, const ConvParameter *conv_param, int start_oh, int end_oh) {
  int units = UP_DIV(conv_param->output_w_, C2NUM);
  int c4 = UP_ROUND(conv_param->input_channel_, C4NUM);
  int line = conv_param->input_channel_ * conv_param->input_w_;

  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;

  for (int b = 0; b < conv_param->output_batch_; b++) {
    const float *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    float *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    float *line0 = buffer;
    float *line1 = buffer + units * c4 * C4NUM;
    float *line2 = buffer + units * c4 * C8NUM;
    float *lines[3] = {line0, line1, line2};
    int oh = start_oh;
    if (oh == 0) {
      // input trans
      ConvDw3x3InitTop(src, lines, conv_param->output_w_, conv_param->input_channel_);
    } else {
      // input trans
      ConvDw3x3InitRow(src + oh * line, lines, conv_param->output_w_, conv_param->input_channel_);
    }
    // dst calc and trans
    ConvDw3x3Line(dst + oh * line, lines, weight_data, bias_data, conv_param->output_w_, conv_param->input_channel_,
                  relu, relu6);
    for (oh = start_oh + 1; oh < end_oh - 1; oh++) {
      // input trans
      ConvDw3x3Row(src + oh * line + line, lines, conv_param->output_w_, conv_param->input_channel_);
      // dst calc and trans
      ConvDw3x3Line(dst + oh * line, lines, weight_data, bias_data, conv_param->output_w_, conv_param->input_channel_,
                    relu, relu6);
    }
    if (oh == conv_param->output_h_ - 1) {
      // input trans
      ConvDw3x3Bottom(lines, conv_param->output_w_, conv_param->input_channel_);
    } else {
      // input trans
      ConvDw3x3Row(src + oh * line + line, lines, conv_param->output_w_, conv_param->input_channel_);
    }
    // dst calc and trans
    ConvDw3x3Line(dst + oh * line, lines, weight_data, bias_data, conv_param->output_w_, conv_param->input_channel_,
                  relu, relu6);
  }
}
#endif

/*conv depthwise indirect buffer fp32 begin*/
bool CheckConvDwUseIndirectBuffer(const ConvParameter *conv_param) {
  bool use_indirect = (conv_param->kernel_h_ == 3 && conv_param->kernel_w_ == 3) ||
                      (conv_param->kernel_h_ == 5 && conv_param->kernel_w_ == 5);
  return use_indirect;
}

void ConvDwInitIndirection(float **indirect_buffer, float *src, float *zero_ptr, const ConvParameter *conv_param,
                           int step_h, int step_w) {
#ifdef ENABLE_AVX
  int div = C8NUM;
#else
  int div = C4NUM;
#endif

  int ic_div = UP_DIV(conv_param->input_channel_, div) * div;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    float **indirect = indirect_buffer + b * conv_param->output_h_ * step_h;
    float *input = src + b * conv_param->input_h_ * conv_param->input_w_ * ic_div;
    for (int oh = 0; oh < conv_param->output_h_; oh++) {
      for (int kh = 0; kh < conv_param->kernel_h_; kh++) {
        int ih = oh * conv_param->stride_h_ + kh * conv_param->dilation_h_ - conv_param->pad_u_;
        if (ih < conv_param->input_h_ && ih >= 0) {
          for (int ow = 0; ow < conv_param->output_w_; ow++) {
            for (int kw = 0; kw < conv_param->kernel_w_; kw++) {
              int iw = ow * conv_param->stride_w_ + kw * conv_param->dilation_w_ - conv_param->pad_l_;
              int index = oh * step_h + ow * step_w * conv_param->kernel_h_ + kw * conv_param->kernel_h_ + kh;
              if (iw < conv_param->input_w_ && iw >= 0) {
                indirect[index] = input + (ih * conv_param->input_w_ + iw) * ic_div;
              } else {
                indirect[index] = zero_ptr;
              }
            }
          }
        } else {
          for (int ow = 0; ow < conv_param->output_w_; ow++) {
            for (int kw = 0; kw < conv_param->kernel_w_; kw++) {
              int index = oh * step_h + ow * step_w * conv_param->kernel_h_ + kw * conv_param->kernel_h_ + kh;
              indirect[index] = zero_ptr;
            }
          }
        }
      }
    }
  }
}

#if !defined(ENABLE_ARM64) && !defined(ENABLE_AVX)
void ConvDwFp32IndirectRow(float *output, float **input, const float *weights, const float *bias, int channels,
                           int output_width, int input_stride, bool relu, bool relu6, int kernel) {
  do {
    float *in[kernel];
    for (int k = 0; k < kernel; k++) {
      in[k] = input[k];
    }
    input = input + input_stride;

    size_t c = channels;
    const float *w = weights;
    float *out = output;
    memcpy(out, bias, channels * sizeof(float));
    for (; c >= C4NUM; c -= C4NUM) {
      for (int i = 0; i < C4NUM; i++) {
        for (int k = 0; k < kernel; k++) {
          out[i] += in[k][i] * w[i + k * C4NUM];
        }
      }
      w += kernel * C4NUM;
      out += C4NUM;
      for (int k = 0; k < kernel; k++) {
        in[k] += C4NUM;
      }
    }
    for (int i = 0; i < c; i++) {
      for (int k = 0; k < kernel; k++) {
        out[i] += in[k][i] * w[i + k * C4NUM];
      }
    }
    if (relu) {
      ReluFp32(output, output, channels);
    }
    if (relu6) {
      Relu6Fp32(output, output, channels);
    }
    output += channels;
  } while (--output_width != 0);
}
#endif

#ifdef ENABLE_ARM64
void ConvDwFp32IndirectRow(float *output, float **input, const float *weights, const float *bias, int channels,
                           int output_width, int input_stride, bool relu, bool relu6, int kernel) {
  if (kernel == 9) {
    ConvDwFp32Indirect3x3(output, input, weights, bias, channels, output_width, input_stride * sizeof(float *), relu,
                          relu6);
  } else if (kernel == 25) {
    ConvDwFp32Indirect5x5(output, input, weights, bias, channels, output_width, input_stride * sizeof(float *), relu,
                          relu6);
  }
}
#endif

#ifdef ENABLE_AVX
void ConvDwFp32IndirectRow(float *output, float **input, const float *weights, const float *bias, int channels,
                           int output_width, int input_stride, bool relu, bool relu6, int kernel) {
  if (kernel == 9) {
    ConvDwFp32Avx3x3(output, input, weights, bias, channels, output_width, input_stride * sizeof(float *), relu, relu6);
  } else if (kernel == 25) {
    ConvDwFp32Avx5x5(output, input, weights, bias, channels, output_width, input_stride * sizeof(float *), relu, relu6);
  }
}
#endif

void ConvDwIndirection(float *output_data, float **indirect_buffer, const float *weight_data, const float *bias_data,
                       float *zero_ptr, const ConvParameter *conv_param, int task_id) {
  int step_w = conv_param->dilation_w_ == 1 ? conv_param->stride_w_ : conv_param->kernel_w_;
  int step_h =
    (conv_param->kernel_h_ * conv_param->kernel_w_) + (conv_param->output_w_ - 1) * step_w * conv_param->kernel_h_;
  int input_stride = conv_param->kernel_h_ * step_w;

  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;

  int h_step = UP_DIV(conv_param->output_h_, conv_param->thread_num_);
  int h_start = h_step * task_id;
  int h_end = MSMIN(h_start + h_step, conv_param->output_h_);

  for (int b = 0; b < conv_param->output_batch_; b++) {
    float **indirect_b = indirect_buffer + b * conv_param->output_h_ * step_h;
    float *outout_b = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    for (int oh = h_start; oh < h_end; oh++) {
      float **indirect = indirect_b + oh * step_h;
      float *output_h = outout_b + oh * conv_param->output_w_ * conv_param->output_channel_;
      if (conv_param->kernel_w_ == 3) {
        ConvDwFp32IndirectRow(output_h, indirect, weight_data, bias_data, conv_param->output_channel_,
                              conv_param->output_w_, input_stride, relu, relu6, 9);
      } else if (conv_param->kernel_w_ == 5) {
        ConvDwFp32IndirectRow(output_h, indirect, weight_data, bias_data, conv_param->output_channel_,
                              conv_param->output_w_, input_stride, relu, relu6, 25);
      }
    }
  }
}
/*conv depthwise indirect buffer fp32 end*/

/*deconv depthwise fp32 begin*/
void DeconvDwBorderPixel(float *dst, const float *src, const float *weight, int height, int width, int in_kh_step,
                         int in_kw_step, int kernel_w_step) {
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

void DeconvDwBorder(float *dst, const float *src, const float *weight, int top, int bottom, int left, int right,
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
      DeconvDwBorderPixel(dst_kernel, src_kernel, weight_kernel, end_kh - start_kh, end_kw - start_kw,
                          sliding->in_kh_step_, sliding->in_kw_step_, conv_param->kernel_w_ * C4NUM);
#endif
      src_kernel += sliding->block_channel_;
    }  // width loop
    src_h += sliding->out_h_step_;
  }  // height loop
}

#ifndef ENABLE_ARM64
void DeconvDwCenter(float *dst, const float *src, const float *weight, int height, int width, int kernel_h,
                    int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step, int in_kh_step,
                    int in_kw_step) {
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

void DeconvDwPost(float *dst, const float *bias, int block_channel, const ConvParameter *conv_param) {
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
void DeconvDwSWFp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                    const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id) {
  const float *src = input_data;
  float *dst = output_data;
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oc = task_id; oc < sliding->c_block_; oc += conv_param->thread_num_) {
      const float *src_data = src + oc * C4NUM;
      float *dst_data = dst + oc * C4NUM;
      const float *weight = weight_data + oc * sliding->kernel_step_;
      const float *bias = bias_data + oc * C4NUM;
      DeconvDwBorder(dst_data, src_data, weight, 0, sliding->top_, 0, conv_param->input_w_, conv_param, sliding);
      DeconvDwBorder(dst_data, src_data, weight, sliding->bottom_, conv_param->input_h_, 0, conv_param->input_w_,
                     conv_param, sliding);
      DeconvDwBorder(dst_data, src_data, weight, sliding->top_, sliding->bottom_, 0, sliding->left_, conv_param,
                     sliding);
      DeconvDwBorder(dst_data, src_data, weight, sliding->top_, sliding->bottom_, sliding->right_, conv_param->input_w_,
                     conv_param, sliding);

      if (sliding->right_ > sliding->left_ && sliding->bottom_ > sliding->top_) {
        int oh_h_start = sliding->top_ * conv_param->stride_h_ - conv_param->pad_u_;
        int oh_w_start = sliding->left_ * conv_param->stride_w_ - conv_param->pad_l_;
        float *out_t = dst_data + oh_h_start * sliding->in_h_step_ + oh_w_start * sliding->block_channel_;
        const float *in_t = src_data + sliding->top_ * sliding->out_h_step_ + sliding->left_ * sliding->block_channel_;

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
        DeconvDwFp32Center(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                           conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_ * sizeof(float),
                           sliding->block_channel_ * sizeof(float), sliding->in_sh_step_ * sizeof(float),
                           sliding->in_sw_step_ * sizeof(float), sliding->in_kh_step_ * sizeof(float),
                           sliding->in_kw_step_ * sizeof(float));
#else
        DeconvDwCenter(out_t, in_t, weight, sliding->bottom_ - sliding->top_, sliding->right_ - sliding->left_,
                       conv_param->kernel_h_, conv_param->kernel_w_, sliding->out_h_step_, sliding->block_channel_,
                       sliding->in_sh_step_, sliding->in_sw_step_, sliding->in_kh_step_, sliding->in_kw_step_);
#endif
      }
      DeconvDwPost(dst_data, bias, sliding->block_channel_, conv_param);
    }  // output C4 loop
    src += sliding->out_step_;
    dst += sliding->in_step_;
  }  // batch loop
  // output nhwc4
}
/*deconv depthwise fp32 end*/
