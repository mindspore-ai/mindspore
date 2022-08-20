/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/conv_depthwise_avx_fp32.h"
#include "nnacl/common_func.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/errorcode.h"
#include "nnacl/fp32/activation_fp32.h"

int ConvDwAVX(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
              const ConvParameter *conv_param, int task_id, ConvDwCalcParam *conv_dw_calc_param) {
  if (conv_param->thread_num_ == 0 || conv_param->dilation_h_ == 0 || conv_param->stride_w_ == 0) {
    return NNACL_ERR;
  }

  int *num_pixels = conv_dw_calc_param->num_pixels_;
  int *out_w_start = conv_dw_calc_param->out_w_start_;
  int first_calc_kw = conv_dw_calc_param->first_calc_kw_;

  int h_step = UP_DIV(conv_param->output_h_, conv_param->thread_num_);
  int h_start = h_step * task_id;
  int h_end = MSMIN(h_start + h_step, conv_param->output_h_);
  bool relu = conv_param->act_type_ == ActType_Relu;
  bool relu6 = conv_param->act_type_ == ActType_Relu6;

  for (int b = 0; b < conv_param->output_batch_; b++) {
    const float *src = input_data + b * conv_param->input_h_ * conv_param->input_w_ * conv_param->input_channel_;
    float *dst = output_data + b * conv_param->output_h_ * conv_param->output_w_ * conv_param->output_channel_;
    for (int oh = h_start; oh < h_end; oh++) {
      int ih_origin = oh * conv_param->stride_h_ - conv_param->pad_u_;
      int start_kh = MSMAX(0, UP_DIV(-ih_origin, conv_param->dilation_h_));
      int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih_origin, conv_param->dilation_h_));
      float *dst_data = dst + oh * conv_param->output_w_ * conv_param->output_channel_;

      bool first_calc_flag = true;
      if (first_calc_kw == -1) {
        for (int ow = 0; ow < conv_param->output_w_; ow++) {
          memcpy(dst_data + ow * conv_param->output_channel_, bias_data,
                 conv_param->output_channel_ * (int)(sizeof(float)));
        }
        first_calc_flag = false;
      }
      for (int kh = start_kh; kh < end_kh; kh++) {
        int ih = ih_origin + conv_param->dilation_h_ * kh;
        int in_sw_step = conv_param->stride_w_ * conv_param->input_channel_;

        const float *src_kh = src + ih * conv_param->input_w_ * conv_param->input_channel_;
        const float *weight_kh = weight_data + kh * conv_param->kernel_w_ * conv_param->output_channel_;

        if (first_calc_flag) {
          int iw_origin = -conv_param->pad_l_ + conv_param->dilation_w_ * first_calc_kw;
          const float *src_kw = src_kh + iw_origin * conv_param->input_channel_;
          ConvDwAVXFp32Row(dst_data, src_kw, weight_kh + first_calc_kw * conv_param->output_channel_,
                           conv_param->output_w_, conv_param->output_channel_, in_sw_step, true, bias_data);
        }
        for (int kw = 0; kw < conv_param->kernel_w_; kw++) {
          if (first_calc_flag && (kw == first_calc_kw)) {
            weight_kh += conv_param->output_channel_;
            first_calc_flag = false;
            continue;
          }
          int iw_origin = (out_w_start[kw] * conv_param->stride_w_) - conv_param->pad_l_ + conv_param->dilation_w_ * kw;
          const float *src_kw = src_kh + iw_origin * conv_param->input_channel_;
          float *dst_w = dst_data + out_w_start[kw] * conv_param->output_channel_;

          ConvDwAVXFp32Row(dst_w, src_kw, weight_kh, num_pixels[kw], conv_param->output_channel_, in_sw_step, false,
                           bias_data);
          weight_kh += conv_param->output_channel_;
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
