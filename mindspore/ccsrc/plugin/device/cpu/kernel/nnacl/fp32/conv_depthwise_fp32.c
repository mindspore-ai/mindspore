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
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/errorcode.h"
#include "nnacl/fp32/activation_fp32.h"

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

int ConvDw(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
           const ConvParameter *conv_param, int task_id) {
  if (conv_param->thread_num_ == 0 || conv_param->dilation_h_ == 0 || conv_param->stride_w_ == 0) {
    return NNACL_ERR;
  }
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
        memcpy(dst_data + ow * conv_param->output_channel_, bias_data,
               conv_param->output_channel_ * (int)(sizeof(float)));
      }
      for (int kh = start_kh; kh < end_kh; kh++) {
        int ih = ih_origin + conv_param->dilation_h_ * kh;

        const float *src_kh = src + ih * conv_param->input_w_ * conv_param->input_channel_;
        const float *dw_weight_kh = weight_data + kh * conv_param->kernel_w_ * conv_param->output_channel_;

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

          ConvDwFp32Row(dst_w, src_kw, dw_weight_kh, num_pixels, conv_param->output_channel_, in_sw_step);
          dw_weight_kh += conv_param->output_channel_;
        }
      }
      if (relu) {
        Fp32Relu(dst_data, conv_param->output_w_ * conv_param->output_channel_, dst_data);
      }
      if (relu6) {
        Fp32Relu6(dst_data, conv_param->output_w_ * conv_param->output_channel_, dst_data);
      }
    }
  }
  return NNACL_OK;
}

#ifdef ENABLE_AVX512
int ConvDwAVX512(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                 const ConvParameter *conv_param, int task_id, ConvDwCalcParam *conv_dw_calc_param) {
  if (conv_param->thread_num_ == 0 || conv_param->dilation_h_ == 0 || conv_param->stride_w_ == 0) {
    return NNACL_ERR;
  }
  int h_step = UP_DIV(conv_param->output_h_, conv_param->thread_num_);
  int h_start = h_step * task_id;
  int h_end = MSMIN(h_start + h_step, conv_param->output_h_);
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;

  int *num_pixels = conv_dw_calc_param->num_pixels_;
  int *out_w_start = conv_dw_calc_param->out_w_start_;
  int first_calc_kw = conv_dw_calc_param->first_calc_kw_;

  for (int b = 0; b < conv_param->output_batch_; b++) {
    const float *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    float *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    for (int oh = h_start; oh < h_end; oh++) {
      float *dst_data = dst + oh * conv_param->output_w_ * conv_param->output_channel_;

      int ih_origin = oh * conv_param->stride_h_ - conv_param->pad_u_;
      int start_kh = MSMAX(0, UP_DIV(-ih_origin, conv_param->dilation_h_));
      int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih_origin, conv_param->dilation_h_));

      bool first_calc_flag = true;
      if (first_calc_kw == -1) {
        first_calc_flag = false;
        for (int ow = 0; ow < conv_param->output_w_; ow++) {
          memcpy(dst_data + ow * conv_param->output_channel_, bias_data,
                 conv_param->output_channel_ * (int)(sizeof(float)));
        }
      }
      for (int kh = start_kh; kh < end_kh; kh++) {
        int ih = ih_origin + conv_param->dilation_h_ * kh;

        const float *src_kh = src + ih * conv_param->input_w_ * conv_param->input_channel_;
        const float *weight_kh = weight_data + kh * conv_param->kernel_w_ * conv_param->output_channel_;

        int in_sw_step = conv_param->stride_w_ * conv_param->input_channel_;

        if (first_calc_flag) {
          int iw_origin = -conv_param->pad_l_ + conv_param->dilation_w_ * first_calc_kw;

          const float *src_kw = src_kh + iw_origin * conv_param->input_channel_;

          ConvDwAVX512Fp32Row(dst_data, src_kw, weight_kh + first_calc_kw * conv_param->output_channel_,
                              conv_param->output_w_, conv_param->output_channel_, in_sw_step, true, bias_data);
        }
        for (int kw = 0; kw < conv_param->kernel_w_; kw++) {
          if (first_calc_flag && (kw == first_calc_kw)) {
            first_calc_flag = false;
            weight_kh += conv_param->output_channel_;
            continue;
          }

          float *dst_w = dst_data + out_w_start[kw] * conv_param->output_channel_;
          int iw_origin = (out_w_start[kw] * conv_param->stride_w_) - conv_param->pad_l_ + conv_param->dilation_w_ * kw;

          const float *src_kw = src_kh + iw_origin * conv_param->input_channel_;

          ConvDwAVX512Fp32Row(dst_w, src_kw, weight_kh, num_pixels[kw], conv_param->output_channel_, in_sw_step, false,
                              bias_data);
          weight_kh += conv_param->output_channel_;
        }
      }
      if (relu) {
        Fp32Relu(dst_data, conv_param->output_w_ * conv_param->output_channel_, dst_data);
      } else if (relu6) {
        Fp32Relu6(dst_data, conv_param->output_w_ * conv_param->output_channel_, dst_data);
      }
    }
  }
  return NNACL_OK;
}
#endif

void InitSlidingParam(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block) {
  if (block == 0) {
    return;
  }
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
  if (conv_param->out_format_ == Format_NC4HW4) {
    // write to nc8hw8
    sliding->out_h_step_ = conv_param->output_w_ * block;
    sliding->out_c_step_ = block * conv_param->output_h_ * conv_param->output_w_;
    sliding->out_w_step_ = block;
    sliding->out_block_step_ = sliding->out_c_step_;
  } else {
    // write to nhwc
    sliding->out_h_step_ = conv_param->output_w_ * sliding->block_channel_;
    sliding->out_c_step_ = block;
    sliding->out_w_step_ = sliding->block_channel_;
    sliding->out_block_step_ = sliding->out_w_step_;
  }
}

void InitSlidingParamConv(SlidingWindowParam *sliding, const ConvParameter *conv_param, int input_block,
                          int weight_block) {
  InitSlidingParam(sliding, conv_param, weight_block);
  AppendSlidingParamConv(sliding, conv_param, input_block, weight_block);
}

void AppendSlidingParamConv(SlidingWindowParam *sliding, const ConvParameter *conv_param, int input_block,
                            int weight_block) {
  if (input_block == 0) {  // is not aligned
    sliding->ic_align_ = conv_param->input_channel_;
  } else {  // 1x1 input is aligned to input_block
    sliding->ic_align_ = UP_DIV(conv_param->input_channel_, input_block) * input_block;
  }
  sliding->in_step_ = conv_param->input_h_ * conv_param->input_w_ * sliding->ic_align_;  // for batch loop
  sliding->in_h_step_ = conv_param->input_w_ * sliding->ic_align_;
  sliding->in_sh_step_ = conv_param->input_w_ * sliding->ic_align_ * conv_param->stride_h_;    // stride H
  sliding->in_sw_step_ = sliding->ic_align_ * conv_param->stride_w_;                           // stride W
  sliding->in_kh_step_ = conv_param->input_w_ * sliding->ic_align_ * conv_param->dilation_h_;  // kernel H
  sliding->in_kw_step_ = sliding->ic_align_ * conv_param->dilation_w_;                         // kernel W
  sliding->kernel_step_ = conv_param->kernel_w_ * conv_param->kernel_h_ * sliding->ic_align_ * weight_block;
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
  if (conv_param->dilation_h_ == 0 || conv_param->dilation_w_ == 0) {
    return;
  }
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
      if (param == NULL) {
        return;
      }
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
  if (conv_param->thread_num_ == 0) {
    return;
  }
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
static void ConvDw3x3RowLeft(const float *src, float *line, int lw, int channel) {
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
    memset(remain_line, 0, 64);
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

static void ConvDw3x3RowMiddle(const float *src, float *line, int lw, int channel) {
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
    memset(remain_line, 0, 64);
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

static void ConvDw3x3RowRight(const float *src, float *line, int lw, int channel) {
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
    memset(remain_line, 0, 64);
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

static void ConvDw3x3RowSingle(const float *src, float *line, int lw, int channel) {
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
    memset(remain_line, 0, 64);
    for (int i = 0; i < channel - ic; i++) {
      float d0 = src[i + ic];
      float d1 = src[i + ic + channel];
      remain_line[i] = d0;
      remain_line[i + 4] = d1;
      remain_line[i + 8] = 0.0f - d1;
    }
  }
}

static void ConvDw3x3InitTop(const float *src, float **lines, int width, int channel) {
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

static void ConvDw3x3InitRow(const float *src, float **lines, int width, int channel) {
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

static void ConvDw3x3Row(const float *src, float **lines, int width, int channel) {
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

static void ConvDw3x3Bottom(float **lines, int width, int channel) {
  float *tmp = lines[0];
  lines[0] = lines[1];
  lines[1] = lines[2];
  lines[2] = tmp;
  int c4 = UP_ROUND(channel, C4NUM);
  memset(tmp, 0, UP_DIV(width, C2NUM) * c4 * C4NUM * sizeof(float));
}

#ifndef ENABLE_ARM64
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
          cur_dst[i] = MS_F32X4_GETI(res0, i);
          cur_dst[ori_channel + i] = MS_F32X4_GETI(res1, i);
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
          cur_dst[i] = MS_F32X4_GETI(res0, i);
        }
      }
    }
    dst += 4;
  }
}
#endif

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
    float **in = input;
    size_t c = (size_t)channels;
    const float *w = weights;
    float *out = output;
    memcpy(out, bias, channels * (int)sizeof(float));
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
      Fp32Relu(output, channels, output);
    }
    if (relu6) {
      Fp32Relu6(output, channels, output);
    }
    output += channels;
    input = input + input_stride;
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
  if (conv_param->thread_num_ == 0) {
    return;
  }
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
  if (conv_param->dilation_h_ == 0 || conv_param->dilation_w_ == 0) {
    return;
  }
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
  if (conv_param->thread_num_ == 0) {
    return;
  }
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

#ifdef ENABLE_AVX
void DepthwiseBorderAvxFp32(float *dst, const float *src, const float *weight, const float *bias, int top, int left,
                            int right, const ConvParameter *conv_param, const SlidingWindowParam *sw_param,
                            const DepthwiseSWKernel kernel, int act_type, int ow_bock, int oc_block) {
  // dw border compate
  int ih = top * conv_param->stride_h_ - conv_param->pad_u_;
  int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));
  int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));
  const float *src_h = src + ih * sw_param->in_h_step_;
  float *dst_kernel = dst + left * sw_param->block_channel_;
  for (int ow = left; ow < right; ow += ow_bock) {
    int iw = ow * conv_param->stride_w_ - conv_param->pad_l_;
    int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));
    int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));
    const float *src_w = src_h + iw * sw_param->block_channel_;
    const float *src_kernel = src_w + start_kh * sw_param->in_kh_step_ + start_kw * sw_param->in_kw_step_;
    const float *weight_kernel = weight + (start_kh * conv_param->kernel_w_ + start_kw) * C8NUM * oc_block;
    kernel(dst_kernel, src_kernel, weight_kernel, bias, end_kh - start_kh, end_kw - start_kw, act_type, ow_bock,
           oc_block, sw_param->block_channel_, sw_param->in_kw_step_, sw_param->in_kh_step_, sw_param->in_sw_step_,
           (conv_param->kernel_w_ - end_kw + start_kw) * C8NUM * oc_block);
    dst_kernel += ow_bock * sw_param->block_channel_;
  }  // width loop
}

void DepthwiseSWAvxFp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                        const ConvParameter *conv_param, const SlidingWindowParam *sw_param, int task_id) {
  int oh_step = UP_DIV(conv_param->output_h_, conv_param->thread_num_);
  int oh_start = oh_step * task_id;
  int oh_end = MSMIN(oh_start + oh_step, conv_param->output_h_);
  if (oh_start >= oh_end) {
    return;
  }
  // depthwise sw in x86 avx instructions
  int oc_tile_ = C8NUM;  // oc in algin to C8NUM in x86_64_avx
  int act_type = 0;
  if (conv_param->act_type_ == ActType_Relu6) {
    act_type += 1;
  }
  if (conv_param->act_type_ == ActType_Relu || conv_param->act_type_ == ActType_Relu6) {
    act_type += 2;
  }
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int output_w = conv_param->output_w_;
  int oc_algin = sw_param->block_channel_;
  int oc_num = sw_param->c_block_;
  int in_step = sw_param->in_step_;
  int out_step = sw_param->out_step_;
  int in_sw_step = sw_param->in_sw_step_;
  int in_kw_step = sw_param->in_kw_step_;
  int in_kh_step = sw_param->in_kh_step_;
  int in_sh_step = sw_param->in_sh_step_;
  int out_right = sw_param->right_;
  int out_left = sw_param->left_;
  int out_top = sw_param->top_;
  int out_bottom = sw_param->bottom_;
  int kernel_step = sw_param->kernel_step_;
  int out_h_step = sw_param->out_h_step_;
  int in_h_start = out_top * conv_param->stride_h_ - conv_param->pad_u_;
  int in_w_start = out_left * conv_param->stride_w_ - conv_param->pad_l_;
  int in_start = in_h_start * sw_param->in_h_step_ + in_w_start * oc_algin;
  const int ow_block_num[4] = {8, 4, 4, 3};
  const DepthwiseSWKernel kernel[4][2] = {{DepthwiseSW1x8Kernel, DepthwiseSW8x8Kernel},
                                          {DepthwiseSW1x16Kernel, DepthwiseSW4x16Kernel},
                                          {DepthwiseSW1x24Kernel, DepthwiseSW4x24Kernel},
                                          {DepthwiseSW1x32Kernel, DepthwiseSW3x32Kernel}};
  for (int b = 0; b < conv_param->output_batch_; b++) {
    for (int oh = oh_start; oh < oh_end; ++oh) {
      float *dst_oh = output_data + oh * out_h_step;
      const float *src_h = input_data + in_start + (oh - out_top) * in_sh_step;
      int oc_block = 0;
      const float *bias = bias_data;
      for (int oc = 0; oc < oc_num; oc += oc_block) {
        oc_block = MSMIN(C4NUM, oc_num - oc);  // 4 3 2 1
        int oc_step = oc * oc_tile_;
        const float *weight = weight_data + oc * kernel_step;
        if (bias != NULL) {
          bias = bias_data + oc_step;
        }
        float *dst_w = dst_oh + oc_step;
        const DepthwiseSWKernel kernel_border = kernel[oc_block - 1][0];
        if (oh < out_top || oh >= out_bottom) {  // oh in up or down border
          DepthwiseBorderAvxFp32(dst_w, input_data + oc_step, weight, bias, oh, 0, output_w, conv_param, sw_param,
                                 kernel_border, act_type, 1, oc_block);
        } else {  // oh in center
          // ow in right
          DepthwiseBorderAvxFp32(dst_w, input_data + oc_step, weight, bias, oh, 0, out_left, conv_param, sw_param,
                                 kernel_border, act_type, 1, oc_block);
          // ow in center
          const float *src_w = src_h + oc_step;
          int ow_block = ow_block_num[oc_block - 1];                 // 8 4 4 3
          for (int ow = out_left; ow < out_right; ow += ow_block) {  // left ~ right
            if (ow_block > out_right - ow) {                         // ow is not enough and process one ow
              ow_block = 1;
            }
            kernel[oc_block - 1][ow_block / ow_block_num[oc_block - 1]](
              dst_w + ow * oc_algin, src_w, weight, bias, kernel_h, kernel_w, act_type, ow_block, oc_block, oc_algin,
              in_kw_step, in_kh_step, in_sw_step, 0);
            src_w += ow_block * in_sw_step;
          }
          // ow in left
          DepthwiseBorderAvxFp32(dst_w, input_data + oc_step, weight, bias, oh, out_right, output_w, conv_param,
                                 sw_param, kernel_border, act_type, 1, oc_block);
        }
      }
    }  // output h loop
    input_data += in_step;
    output_data += out_step;
  }  // batch loop
}

#ifdef ENABLE_DEBUG
void DepthwiseSWWxKKernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                          size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                          size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  __m256 dst_data[12];
  __m256 src_data;
  const float *src_kh[12];
  const float *src_kw[12];
  __m256 weight_data[4];
  for (int i = 0; i < ow_block; ++i) {
    if (bias != NULL) {
      for (int j = 0; j < oc_block; ++j) {
        dst_data[i * oc_block + j] = _mm256_loadu_ps(bias + j * 8);
      }
    } else {
      for (int j = 0; j < oc_block; ++j) {
        dst_data[i * oc_block + j] = _mm256_set1_ps(0.0f);
      }
    }
    src_kh[i] = src + i * in_sw_step;
    src_kw[i] = NULL;
  }
  const float *weight_kernel = weight;
  for (int kh = 0; kh < kernel_h; kh++) {
    for (int i = 0; i < ow_block; ++i) {
      src_kw[i] = src_kh[i];
    }
    for (int kw = 0; kw < kernel_w; kw++) {
      for (int j = 0; j < oc_block; ++j) {
        weight_data[j] = _mm256_loadu_ps(weight_kernel + j * C8NUM);
      }
      for (int i = 0; i < ow_block; ++i) {  // loop ow
        for (int j = 0; j < oc_block; ++j) {
          src_data = _mm256_loadu_ps(src_kw[i] + j * C8NUM);
          dst_data[i * oc_block + j] += src_data * weight_data[j];
        }
      }
      for (int i = 0; i < ow_block; ++i) {
        src_kw[i] += in_kw_step;  // ic8 * dilation_w
      }
      weight_kernel += oc_block * C8NUM;
    }  // kernel_w loop
    weight_kernel += kw_remainder;
    for (int i = 0; i < ow_block; ++i) {
      src_kh[i] += in_kh_step;  //
    }
  }  // kernel_h loop
  // add bias and relu
  for (int i = 0; i < ow_block; ++i) {
    for (int j = 0; j < oc_block; ++j) {
      if (0x1 & act_flag) {  // relu6
        dst_data[i * oc_block + j] = _mm256_min_ps(dst_data[i * oc_block + j], _mm256_set1_ps(6.0f));
      }
      if (0x2 & act_flag) {  // relu
        dst_data[i * oc_block + j] = _mm256_max_ps(dst_data[i * oc_block + j], _mm256_set1_ps(0.0f));
      }
      _mm256_storeu_ps(dst + i * oc_algin + j * C8NUM, dst_data[i * oc_block + j]);
    }
  }
}
#endif

void DepthwiseSW3x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_sw_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  oc_algin *= sizeof(float);
  kw_remainder *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "vmovups 0x60(%2), %%ymm3\n"
    "vmovups (%2), %%ymm4\n"
    "vmovups 0x20(%2), %%ymm5\n"
    "vmovups 0x40(%2), %%ymm6\n"
    "vmovups 0x60(%2), %%ymm7\n"
    "vmovups (%2), %%ymm8\n"
    "vmovups 0x20(%2), %%ymm9\n"
    "vmovups 0x40(%2), %%ymm10\n"
    "vmovups 0x60(%2), %%ymm11\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "vxorps %%ymm8, %%ymm8, %%ymm8\n"
    "vxorps %%ymm9, %%ymm9, %%ymm9\n"
    "vxorps %%ymm10, %%ymm10, %%ymm10\n"
    "vxorps %%ymm11, %%ymm11, %%ymm11\n"
    "1:\n"              // LoopH
    "movq %4, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "2:\n"              // LoopW

    "vmovups (%1), %%ymm12\n"
    "vmovups (%%rcx), %%ymm13\n"
    "vmovups (%%rcx, %7), %%ymm14\n"
    "vmovups (%%rcx, %7, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"

    "vmovups 0x20(%1), %%ymm12\n"
    "vmovups 0x20(%%rcx), %%ymm13\n"
    "vmovups 0x20(%%rcx, %7), %%ymm14\n"
    "vmovups 0x20(%%rcx, %7, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm9\n"

    "vmovups 0x40(%1), %%ymm12\n"
    "vmovups 0x40(%%rcx), %%ymm13\n"
    "vmovups 0x40(%%rcx, %7), %%ymm14\n"
    "vmovups 0x40(%%rcx, %7, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm10\n"

    "vmovups 0x60(%1), %%ymm12\n"
    "vmovups 0x60(%%rcx), %%ymm13\n"
    "vmovups 0x60(%%rcx, %7), %%ymm14\n"
    "vmovups 0x60(%%rcx, %7, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm11\n"
    "addq $128, %1\n"

    "addq %5, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg 2b\n"

    "addq %6, %0\n"  // in_kh_step
    "addq %8, %1\n"
    "dec %3\n"
    "jg 1b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step),  // 5
      "r"(in_kh_step), "r"(in_sw_step), "r"(kw_remainder)                               // 8
    : "%rcx", "%rsi", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",
      "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");

  asm volatile(
    "and $0x3, %%eax\n"
    "je 0f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm5, %%ymm5\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"
    "vmaxps %%ymm12, %%ymm8, %%ymm8\n"
    "vmaxps %%ymm12, %%ymm9, %%ymm9\n"
    "vmaxps %%ymm12, %%ymm10, %%ymm10\n"
    "vmaxps %%ymm12, %%ymm11, %%ymm11\n"

    "and $0x1, %%eax\n"
    "je 0f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm5, %%ymm5\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"
    "vminps %%ymm14, %%ymm8, %%ymm8\n"
    "vminps %%ymm14, %%ymm9, %%ymm9\n"
    "vminps %%ymm14, %%ymm10, %%ymm10\n"
    "vminps %%ymm14, %%ymm11, %%ymm11\n"

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%2)\n"
    "vmovups %%ymm2, 0x40(%2)\n"
    "vmovups %%ymm3, 0x60(%2)\n"
    "vmovups %%ymm4, (%2, %1, 1)\n"
    "vmovups %%ymm5, 0x20(%2, %1, 1)\n"
    "vmovups %%ymm6, 0x40(%2, %1, 1)\n"
    "vmovups %%ymm7, 0x60(%2, %1, 1)\n"
    "vmovups %%ymm8, (%2, %1, 2)\n"
    "vmovups %%ymm9, 0x20(%2, %1, 2)\n"
    "vmovups %%ymm10, 0x40(%2, %1, 2)\n"
    "vmovups %%ymm11, 0x60(%2, %1, 2)\n"
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst)
    : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm14");
}

void DepthwiseSW1x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  oc_algin *= sizeof(float);
  kw_remainder *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "vmovups 0x60(%2), %%ymm3\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "1:\n"              // LoopH
    "movq %4, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "2:\n"              // Loopw
    "vmovups (%%rcx), %%ymm4\n"
    "vmovups 0x20(%%rcx), %%ymm5\n"
    "vmovups 0x40(%%rcx), %%ymm6\n"
    "vmovups 0x60(%%rcx), %%ymm7\n"
    // Weight data is loaded directly from memory instead of into registers for calculation.
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm5, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm6, %%ymm2\n"
    "vfmadd231ps 0x60(%1), %%ymm7, %%ymm3\n"
    "addq $128, %1\n"

    "addq %5, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg 2b\n"

    "addq %6, %0\n"  // in_kh_step
    "addq %7, %1\n"
    "dec %3\n"
    "jg 1b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step),  // 5
      "r"(in_kh_step), "r"(kw_remainder)                                                // 7
    : "%rcx", "%rsi", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7");

  asm volatile(
    "and $0x3, %%eax\n"
    "je 0f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"

    "and $0x1, %%eax\n"
    "je 0f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%2)\n"
    "vmovups %%ymm2, 0x40(%2)\n"
    "vmovups %%ymm3, 0x60(%2)\n"
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst)
    : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm12", "%ymm14");
}

void DepthwiseSW4x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  in_sw_step *= sizeof(float);
  kw_remainder *= sizeof(float);
  size_t src_3_step = 3 * in_sw_step;
  float *dst_3 = dst + 3 * oc_algin;
  oc_algin *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    // We need to copy ymm0 to ymm3 to reduce IO time, but unfortunately I didn't find the corresponding instruction.
    "vmovups (%2), %%ymm3\n"
    "vmovups 0x20(%2), %%ymm4\n"
    "vmovups 0x40(%2), %%ymm5\n"
    "vmovups (%2), %%ymm6\n"
    "vmovups 0x20(%2), %%ymm7\n"
    "vmovups 0x40(%2), %%ymm8\n"
    "vmovups (%2), %%ymm9\n"
    "vmovups 0x20(%2), %%ymm10\n"
    "vmovups 0x40(%2), %%ymm11\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "vxorps %%ymm8, %%ymm8, %%ymm8\n"
    "vxorps %%ymm9, %%ymm9, %%ymm9\n"
    "vxorps %%ymm10, %%ymm10, %%ymm10\n"
    "vxorps %%ymm11, %%ymm11, %%ymm11\n"
    "1:\n"              // LoopH
    "movq %4, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "2:\n"              // LoopW
    "vmovups (%1), %%ymm12\n"
    "vmovups (%%rcx), %%ymm13\n"
    "vmovups (%%rcx, %7, 1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm3\n"
    "vmovups (%%rcx, %7, 2), %%ymm15\n"
    "vmovups (%%rcx, %9), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm9\n"

    "vmovups 0x20(%1), %%ymm12\n"
    "vmovups 0x20(%%rcx), %%ymm13\n"
    "vmovups 0x20(%%rcx, %7, 1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vmovups 0x20(%%rcx, %7, 2), %%ymm15\n"
    "vmovups 0x20(%%rcx, %9), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm10\n"

    "vmovups 0x40(%1), %%ymm12\n"
    "vmovups 0x40(%%rcx), %%ymm13\n"
    "vmovups 0x40(%%rcx, %7, 1), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm5\n"
    "vmovups 0x40(%%rcx, %7, 2), %%ymm15\n"
    "vmovups 0x40(%%rcx, %9), %%ymm13\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm8\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm11\n"

    "addq $96, %1\n"
    "addq %5, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg 2b\n"

    "addq %6, %0\n"  // in_kh_step
    "addq %8, %1\n"  // border in sw need to add remainder data
    "dec %3\n"
    "jg 1b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step),  // 5
      "r"(in_kh_step), "r"(in_sw_step), "r"(kw_remainder), "r"(src_3_step)              // 9
    : "%rcx", "%rsi", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9",
      "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");

  asm volatile(
    "and $0x3, %%eax\n"
    "je 0f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm5, %%ymm5\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"
    "vmaxps %%ymm12, %%ymm8, %%ymm8\n"
    "vmaxps %%ymm12, %%ymm9, %%ymm9\n"
    "vmaxps %%ymm12, %%ymm10, %%ymm10\n"
    "vmaxps %%ymm12, %%ymm11, %%ymm11\n"

    "and $0x1, %%eax\n"
    "je 0f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm5, %%ymm5\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"
    "vminps %%ymm14, %%ymm8, %%ymm8\n"
    "vminps %%ymm14, %%ymm9, %%ymm9\n"
    "vminps %%ymm14, %%ymm10, %%ymm10\n"
    "vminps %%ymm14, %%ymm11, %%ymm11\n"

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%2)\n"
    "vmovups %%ymm2, 0x40(%2)\n"
    "vmovups %%ymm3, (%2, %1, 1)\n"
    "vmovups %%ymm4, 0x20(%2, %1, 1)\n"
    "vmovups %%ymm5, 0x40(%2, %1, 1)\n"
    "vmovups %%ymm6, (%2, %1, 2)\n"
    "vmovups %%ymm7, 0x20(%2, %1, 2)\n"
    "vmovups %%ymm8, 0x40(%2, %1, 2)\n"
    "vmovups %%ymm9, (%3)\n"  // dst+3
    "vmovups %%ymm10, 0x20(%3)\n"
    "vmovups %%ymm11, 0x40(%3)\n"
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst), "r"(dst_3)
    : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm14");
}

void DepthwiseSW1x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  oc_algin *= sizeof(float);
  kw_remainder *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "vmovups 0x40(%2), %%ymm2\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "1:\n"              // LoopH
    "movq %4, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "2:\n"              // Loopw
    "vmovups (%%rcx), %%ymm4\n"
    "vmovups 0x20(%%rcx), %%ymm5\n"
    "vmovups 0x40(%%rcx), %%ymm6\n"
    // Weight data is loaded directly from memory instead of into registers for calculation.
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm5, %%ymm1\n"
    "vfmadd231ps 0x40(%1), %%ymm6, %%ymm2\n"
    "addq $96, %1\n"

    "addq %5, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg 2b\n"

    "addq %6, %0\n"  // in_kh_step
    "addq %7, %1\n"  // kw_remainder
    "dec %3\n"
    "jg 1b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step),  // 5
      "r"(in_kh_step), "r"(kw_remainder)                                                // 7
    : "%rcx", "%rsi", "%ymm0", "%ymm1", "%ymm2", "%ymm4", "%ymm5", "%ymm6");

  asm volatile(
    "and $0x3, %%eax\n"
    "je 0f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"

    "and $0x1, %%eax\n"
    "je 0f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%2)\n"
    "vmovups %%ymm2, 0x40(%2)\n"
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst)
    : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm12", "%ymm14");
}

void DepthwiseSW4x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  in_sw_step *= sizeof(float);
  kw_remainder *= sizeof(float);
  size_t src_3_step = 3 * in_sw_step;
  float *dst_3 = dst + 3 * oc_algin;
  oc_algin *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    // We need to copy ymm0 to ymm3 to reduce IO time, but unfortunately I didn't find the corresponding instruction.
    "vmovups (%2), %%ymm3\n"
    "vmovups 0x20(%2), %%ymm4\n"
    "vmovups (%2), %%ymm6\n"
    "vmovups 0x20(%2), %%ymm7\n"
    "vmovups (%2), %%ymm9\n"
    "vmovups 0x20(%2), %%ymm10\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "vxorps %%ymm9, %%ymm9, %%ymm9\n"
    "vxorps %%ymm10, %%ymm10, %%ymm10\n"
    "1:\n"              // LoopH
    "movq %4, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "2:\n"              // LoopW
    "vmovups (%1), %%ymm12\n"
    "vmovups (%%rcx), %%ymm13\n"
    "vmovups (%%rcx, %7, 1), %%ymm14\n"
    "vmovups (%%rcx, %7, 2), %%ymm15\n"
    "vmovups (%%rcx, %9), %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm2, %%ymm9\n"

    "vmovups 0x20(%1), %%ymm12\n"
    "vmovups 0x20(%%rcx), %%ymm13\n"
    "vmovups 0x20(%%rcx, %7, 1), %%ymm14\n"
    "vmovups 0x20(%%rcx, %7, 2), %%ymm15\n"
    "vmovups 0x20(%%rcx, %9), %%ymm2\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm7\n"
    "vfmadd231ps %%ymm12, %%ymm2, %%ymm10\n"

    "addq $64, %1\n"
    "addq %5, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg 2b\n"

    "addq %6, %0\n"  // in_kh_step
    "addq %8, %1\n"  // border in sw need to add remainder data
    "dec %3\n"
    "jg 1b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step),  // 5
      "r"(in_kh_step), "r"(in_sw_step), "r"(kw_remainder), "r"(src_3_step)              // 9
    : "%rcx", "%rsi", "%ymm0", "%ymm1", "%ymm3", "%ymm4", "%ymm6", "%ymm7", "%ymm9", "%ymm10", "%ymm12", "%ymm13",
      "%ymm14", "%ymm15");

  asm volatile(
    "and $0x3, %%eax\n"
    "je 0f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"
    "vmaxps %%ymm12, %%ymm9, %%ymm9\n"
    "vmaxps %%ymm12, %%ymm10, %%ymm10\n"

    "and $0x1, %%eax\n"
    "je 0f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"
    "vminps %%ymm14, %%ymm9, %%ymm9\n"
    "vminps %%ymm14, %%ymm10, %%ymm10\n"

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%2)\n"
    "vmovups %%ymm3, (%2, %1, 1)\n"
    "vmovups %%ymm4, 0x20(%2, %1, 1)\n"
    "vmovups %%ymm6, (%2, %1, 2)\n"
    "vmovups %%ymm7, 0x20(%2, %1, 2)\n"
    "vmovups %%ymm9, (%3)\n"  // dst+3
    "vmovups %%ymm10, 0x20(%3)\n"
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst), "r"(dst_3)
    : "%ecx", "%ymm0", "%ymm1", "%ymm3", "%ymm4", "%ymm6", "%ymm7", "%ymm9", "%ymm10", "%ymm12", "%ymm14");
}

void DepthwiseSW1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  oc_algin *= sizeof(float);
  kw_remainder *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "vmovups 0x20(%2), %%ymm1\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "1:\n"              // LoopH
    "movq %4, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "2:\n"              // Loopw
    "vmovups (%%rcx), %%ymm4\n"
    "vmovups 0x20(%%rcx), %%ymm5\n"
    // Weight data is loaded directly from memory instead of into registers for calculation.
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "vfmadd231ps 0x20(%1), %%ymm5, %%ymm1\n"
    "addq $64, %1\n"

    "addq %5, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg 2b\n"

    "addq %6, %0\n"  // in_kh_step
    "addq %7, %1\n"  // kw_remainder
    "dec %3\n"
    "jg 1b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step),  // 5
      "r"(in_kh_step), "r"(kw_remainder)                                                // 7
    : "%rcx", "%rsi", "%ymm0", "%ymm1", "%ymm4", "%ymm5");

  asm volatile(
    "and $0x3, %%eax\n"
    "je 0f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"

    "and $0x1, %%eax\n"
    "je 0f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, 0x20(%2)\n"
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst)
    : "%ecx", "%ymm0", "%ymm1", "%ymm12", "%ymm14");
}

void DepthwiseSW8x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                          size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                          size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_sw_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  kw_remainder *= sizeof(float);
  size_t src_3_step = 3 * in_sw_step;
  float *dst_3 = dst + 3 * oc_algin;
  float *dst_5 = dst + 5 * oc_algin;
  oc_algin *= sizeof(float);
  asm volatile(
    "cmpq $0, %0\n"
    "je 0f\n"
    "vmovups (%0), %%ymm0\n"
    "vmovups (%0), %%ymm1\n"
    "vmovups (%0), %%ymm2\n"
    "vmovups (%0), %%ymm3\n"
    "vmovups (%0), %%ymm4\n"
    "vmovups (%0), %%ymm5\n"
    "vmovups (%0), %%ymm6\n"
    "vmovups (%0), %%ymm7\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "1:\n"
    :
    : "r"(bias)
    : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7");

  asm volatile(
    "LoopH:\n"
    "movq %3, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "LoopW:\n"
    "movq %%rcx, %%rax\n"
    "vmovups (%1), %%ymm12\n"
    "vmovups (%%rax), %%ymm13\n"
    "vmovups (%%rax, %6), %%ymm14\n"
    "vmovups (%%rax, %6, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm0\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm1\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm2\n"
    "addq %7, %%rax\n"
    "vmovups (%%rax), %%ymm13\n"
    "vmovups (%%rax, %6), %%ymm14\n"
    "vmovups (%%rax, %6, 2), %%ymm15\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm3\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm4\n"
    "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"
    "addq %7, %%rax\n"
    "vmovups (%%rax), %%ymm13\n"
    "vmovups (%%rax, %6), %%ymm14\n"
    "vfmadd231ps %%ymm12, %%ymm13, %%ymm6\n"
    "vfmadd231ps %%ymm12, %%ymm14, %%ymm7\n"

    "addq $32, %1\n"
    "addq %4, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg LoopW\n"

    "addq %5, %0\n"  // in_kh_step
    "addq %8, %1\n"  // border in sw need to add remainder data
    "dec %2\n"
    "jg LoopH\n"
    :
    : "r"(src), "r"(weight), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step), "r"(in_kh_step),  // 5
      "r"(in_sw_step), "r"(src_3_step), "r"(kw_remainder)                                     // 8
    : "%rcx", "%rsi", "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm12",
      "%ymm13", "%ymm14", "%ymm15");

  asm volatile(
    "and $0x3, %%eax\n"
    "je Write\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"
    "vmaxps %%ymm12, %%ymm1, %%ymm1\n"
    "vmaxps %%ymm12, %%ymm2, %%ymm2\n"
    "vmaxps %%ymm12, %%ymm3, %%ymm3\n"
    "vmaxps %%ymm12, %%ymm4, %%ymm4\n"
    "vmaxps %%ymm12, %%ymm5, %%ymm5\n"
    "vmaxps %%ymm12, %%ymm6, %%ymm6\n"
    "vmaxps %%ymm12, %%ymm7, %%ymm7\n"

    "and $0x1, %%eax\n"
    "je Write\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"
    "vminps %%ymm14, %%ymm1, %%ymm1\n"
    "vminps %%ymm14, %%ymm2, %%ymm2\n"
    "vminps %%ymm14, %%ymm3, %%ymm3\n"
    "vminps %%ymm14, %%ymm4, %%ymm4\n"
    "vminps %%ymm14, %%ymm5, %%ymm5\n"
    "vminps %%ymm14, %%ymm6, %%ymm6\n"
    "vminps %%ymm14, %%ymm7, %%ymm7\n"

    "Write:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    "vmovups %%ymm1, (%2, %1)\n"
    "vmovups %%ymm2, (%2, %1, 2)\n"
    "vmovups %%ymm3, (%3)\n"  // dst_3
    "vmovups %%ymm4, (%2, %1, 4)\n"
    "vmovups %%ymm5, (%4)\n"  // dst_5
    "vmovups %%ymm6, (%4, %1, 1)\n"
    "vmovups %%ymm7, (%4, %1, 2)\n"
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst), "r"(dst_3), "r"(dst_5)
    : "%ecx", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm12", "%ymm14");
}

void DepthwiseSW1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                          size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                          size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder) {
  in_kh_step *= sizeof(float);
  in_kw_step *= sizeof(float);
  oc_algin *= sizeof(float);
  kw_remainder *= sizeof(float);
  asm volatile(
    "cmpq $0, %2\n"
    "je 0f\n"
    "vmovups (%2), %%ymm0\n"
    "jmp 1f\n"
    "0:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "1:\n"              // LoopH
    "movq %4, %%rsi\n"  // width
    "movq %0, %%rcx\n"  // src_h
    "2:\n"              // Loopw
    "vmovups (%%rcx), %%ymm4\n"
    // Weight data is loaded directly from memory instead of into registers for calculation.
    "vfmadd231ps (%1), %%ymm4, %%ymm0\n"
    "addq $32, %1\n"

    "addq %5, %%rcx\n"  // in_kw_step
    "dec %%rsi\n"
    "jg 2b\n"

    "addq %6, %0\n"  // in_kh_step
    "addq %7, %1\n"  // kw_remainder
    "dec %3\n"
    "jg 1b\n"
    :
    : "r"(src), "r"(weight), "r"(bias), "r"(kernel_h), "r"(kernel_w), "r"(in_kw_step),  // 5
      "r"(in_kh_step), "r"(kw_remainder)                                                // 7
    : "%rcx", "%rsi", "%ymm0", "%ymm4");

  asm volatile(
    "and $0x3, %%eax\n"
    "je 0f\n"
    // Relu
    "vxorps %%ymm12, %%ymm12, %%ymm12\n"
    "vmaxps %%ymm12, %%ymm0, %%ymm0\n"

    "and $0x1, %%eax\n"
    "je 0f\n"
    // relu6
    "mov $0x40C00000, %%ecx\n"
    "vmovd %%ecx, %%xmm14\n"
    "vpermps %%ymm14, %%ymm12, %%ymm14\n"
    "vminps %%ymm14, %%ymm0, %%ymm0\n"

    "0:\n"
    "vmovups %%ymm0, (%2)\n"  // dst_0
    :
    : "a"(act_flag), "r"(oc_algin), "r"(dst)
    : "%ecx", "%ymm0", "%ymm12", "%ymm14");
}
#endif
