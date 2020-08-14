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

#include "nnacl/int8/deconv.h"
#include "nnacl/int8/matmul_int8.h"

int DeConvInt8(const int8_t *input, const int8_t *weight, int32_t *output, size_t row8, size_t col8, size_t deep,
               ConvParameter *conv_param) {
  MatMulInt8(input, weight, output, row8, col8, deep, conv_param->conv_quant_arg_.input_quant_args_[0].zp_,
             conv_param->conv_quant_arg_.filter_quant_args_[0].zp_);
  return NNACL_OK;
}

int DeConvPostInt8(const int32_t *src, const int32_t *bias, int32_t *tmp, int8_t *out, int output_channel,
                   ConvParameter *conv_param) {
  /* row8x8-major(ih*iw x oc*kh*kw)  ->  row8x8-major(oh*ow x oc) */
  size_t input_plane = conv_param->input_w_ * conv_param->input_h_;
  size_t kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  size_t output_plane = conv_param->output_w_ * conv_param->output_h_;
  int oc8 = UP_DIV(output_channel, C8NUM);
  int in_plane8 = UP_ROUND(input_plane, 8);

  for (int c = 0; c < oc8; c++) {
    int32_t *dst_ptr = tmp + c * output_plane * C8NUM;
    const int32_t *src_ptr = src + c * in_plane8 * kernel_plane * C8NUM;
    memset(dst_ptr, 0, output_plane * C8NUM * sizeof(int32_t));

    for (int ih = 0; ih < conv_param->input_h_; ih++) {
      for (int iw = 0; iw < conv_param->input_w_; iw++) {
        int oh = ih * conv_param->stride_h_ - conv_param->pad_h_;
        int ow = iw * conv_param->stride_w_ - conv_param->pad_w_;

        int kh_start = MSMAX(0, UP_DIV(-oh, conv_param->dilation_h_));
        int kh_end = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->output_h_ - oh, conv_param->dilation_h_));
        int kw_start = MSMAX(0, UP_DIV(-ow, conv_param->dilation_w_));
        int kw_end = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->output_w_ - ow, conv_param->dilation_w_));
        for (int kh = kh_start; kh < kh_end; kh++) {
          for (int kw = kw_start; kw < kw_end; kw++) {
            int src_index = ih * conv_param->input_w_ * C8NUM + iw * C8NUM +
                            kh * input_plane * conv_param->kernel_w_ * C8NUM + kw * input_plane * C8NUM;
            int dst_index = oh * conv_param->output_w_ * C8NUM + ow * C8NUM +
                            kh * conv_param->dilation_h_ * conv_param->output_w_ * C8NUM +
                            kw * conv_param->dilation_w_ * C8NUM;
            for (int i = 0; i < C8NUM; i++) {
              dst_ptr[dst_index + i] += src_ptr[src_index + i];
            }
          } /*kw*/
        }   /*kh*/
      }     /*iw*/
    }       /*ih*/
  }         /*oc8*/

  PostFuncInt8(tmp, bias, out, output_channel, output_plane, UP_ROUND(output_plane, 8),
               conv_param->conv_quant_arg_.quant_multiplier_[0], conv_param->conv_quant_arg_.left_shift_[0],
               conv_param->conv_quant_arg_.right_shift_[0], conv_param->conv_quant_arg_.output_quant_args_[0].zp_,
               conv_param->conv_quant_arg_.out_act_min_[0], conv_param->conv_quant_arg_.out_act_max_[0]);
  return NNACL_OK;
}
