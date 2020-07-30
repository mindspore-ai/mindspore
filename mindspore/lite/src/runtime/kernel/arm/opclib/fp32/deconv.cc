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

#include "src/runtime/kernel/arm/opclib/fp32/deconv.h"

void PackDeConvWeightFp32(const float *weight, float *dst, int input_channel, int output_channel, int plane) {
  /* ichwoc(nhwc)  ->  oc4 * h * w * incUP4 * 4 */
  int ic_up4 = UP_ROUND(input_channel, C4NUM);
  for (int oc = 0; oc < output_channel; oc++) {
    int oc4div = oc / C4NUM;
    int oc4mod = oc % C4NUM;
    for (int ic = 0; ic < input_channel; ic++) {
      for (int hw = 0; hw < plane; hw++) {
        int src_index = ic * plane * output_channel + hw * output_channel + oc;
        int dst_index = oc4div * ic_up4 * plane * C4NUM + hw * ic_up4 * C4NUM + ic * C4NUM + oc4mod;
        dst[dst_index] = weight[src_index];
      }
    }
  }
  return;
}

int DeConvFp32(const float *input, const float *weight, float *output, float *tmp_buffer,
               StrassenMatMulParameter matmul_param) {
  return StrassenMatmul(input, weight, output, &matmul_param, FP32_STRASSEN_MAX_RECURSION, 0, tmp_buffer);
}

int DeConvPostFp32C8x8(const float *src, float *tmp, const float *bias, float *dst, int output_channel,
                       ConvParameter *conv_param) {
  /* row8x8-major(ih*iw x oc*kh*kw)  ->  row8-major(oh*ow x oc) */
  size_t input_plane = conv_param->input_w_ * conv_param->input_h_;
  size_t kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  size_t output_plane = conv_param->output_w_ * conv_param->output_h_;
  int oc8 = UP_DIV(output_channel, C8NUM);
  int in_plane8 = UP_ROUND(input_plane, C8NUM);

  for (int c = 0; c < oc8; c++) {
    float *dst_ptr = tmp + c * output_plane * C8NUM;
    const float *src_ptr = src + c * in_plane8 * kernel_plane * C8NUM;
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
                            kh * in_plane8 * conv_param->kernel_w_ * C8NUM + kw * in_plane8 * C8NUM;
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

  PostConvFuncFp32C8(tmp, dst, bias, output_channel, output_plane, conv_param->output_channel_, conv_param->is_relu_,
                     conv_param->is_relu6_);
  return OPCLIB_OK;
}

int DeConvPostFp32C4(const float *src, float *tmp_c4, float *dst, const float *bias, int output_channel,
                     int input_plane, int kernel_plane, int output_plane, ConvParameter *conv_param) {
  int oc4 = UP_DIV(output_channel, C4NUM);
  for (int c = 0; c < oc4; c++) {
    float *dst_ptr = tmp_c4 + c * output_plane * C4NUM;
    const float *src_ptr = src + c * input_plane * kernel_plane * C4NUM;
    memset(dst_ptr, 0, output_plane * C4NUM * sizeof(float));

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
            int src_index = ih * conv_param->input_w_ * C4NUM + iw * C4NUM +
                            kh * input_plane * conv_param->kernel_w_ * C4NUM + kw * input_plane * C4NUM;
            int dst_index = oh * conv_param->output_w_ * C4NUM + ow * C4NUM +
                            kh * conv_param->dilation_h_ * conv_param->output_w_ * C4NUM +
                            kw * conv_param->dilation_w_ * C4NUM;
            for (int i = 0; i < C4NUM; i++) {
              dst_ptr[dst_index + i] += src_ptr[src_index + i];
            }
          } /*kw*/
        }   /*kh*/
      }     /*iw*/
    }       /*ih*/
  }         /*oc4*/

  PostConvFuncFp32C4(tmp_c4, dst, bias, output_channel, output_plane, conv_param->output_channel_, conv_param->is_relu_,
                     conv_param->is_relu6_);
  return OPCLIB_OK;
}
