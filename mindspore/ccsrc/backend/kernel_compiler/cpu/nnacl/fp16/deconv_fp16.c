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

#include "nnacl/fp16/deconv_fp16.h"

int DeConvPostFp16(const float16_t *src, float16_t *tmp, const float16_t *bias, float16_t *dst, int output_channel,
                   const ConvParameter *conv_param) {
  /* row8x8-major(ih*iw x oc*kh*kw)  ->  row8-major(oh*ow x oc) */
  size_t input_plane = conv_param->input_w_ * conv_param->input_h_;
  size_t kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  size_t output_plane = conv_param->output_w_ * conv_param->output_h_;
  int oc8 = UP_ROUND(output_channel, C8NUM);
  int in_plane16 = UP_ROUND(input_plane, 16);
  int src_iw_stride = C8NUM;
  int src_ih_stride = conv_param->input_w_ * C8NUM;
  int src_kw_stride = in_plane16 * C8NUM;
  int src_kh_stride = in_plane16 * conv_param->kernel_w_ * C8NUM;
  int dst_oh_stride = conv_param->output_w_ * C8NUM;
  int dst_ow_stride = C8NUM;
  int dst_kh_stride = conv_param->dilation_h_ * conv_param->output_w_ * C8NUM;
  int dst_kw_stride = conv_param->dilation_w_ * C8NUM;

  for (int c = 0; c < oc8; c += 8) {
    float16_t *dst_ptr = tmp + c * output_plane;
    const float16_t *src_ptr = src + c * in_plane16 * kernel_plane;
    memset(dst_ptr, 0, output_plane * C8NUM * sizeof(float16_t));

    for (int ih = 0; ih < conv_param->input_h_; ih++) {
      for (int iw = 0; iw < conv_param->input_w_; iw++) {
        int oh = ih * conv_param->stride_h_ - conv_param->pad_u_;
        int ow = iw * conv_param->stride_w_ - conv_param->pad_l_;

        int kh_start = MSMAX(0, UP_DIV(-oh, conv_param->dilation_h_));
        int kh_end = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->output_h_ - oh, conv_param->dilation_h_));
        int kw_start = MSMAX(0, UP_DIV(-ow, conv_param->dilation_w_));
        int kw_end = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->output_w_ - ow, conv_param->dilation_w_));
        for (int kh = kh_start; kh < kh_end; kh++) {
          for (int kw = kw_start; kw < kw_end; kw++) {
            int src_index = ih * src_ih_stride + iw * src_iw_stride + kh * src_kh_stride + kw * src_kw_stride;
            int dst_index = oh * dst_oh_stride + ow * dst_ow_stride + kh * dst_kh_stride + kw * dst_kw_stride;
            float16_t *tmp_dst = dst_ptr + dst_index;
            const float16_t *tmp_src = src_ptr + src_index;
#ifdef ENABLE_ARM64
            asm volatile(
              "mov x0, %[tmp_src] \n"
              "mov x1, %[tmp_dst] \n"

              "ld1 {v0.8h}, [x0] \n"
              "ld1 {v1.8h}, [x1] \n"

              "fadd v0.8h, v0.8h, v1.8h \n"

              "st1 {v0.8h}, [x1] \n"

              :
              : [ tmp_src ] "r"(tmp_src), [ tmp_dst ] "r"(tmp_dst)
              : "x0", "x1", "v0", "v1");
#else
            for (int i = 0; i < C8NUM; i++) {
              tmp_dst[i] += tmp_src[i];
            }
#endif
          } /*kw*/
        }   /*kh*/
      }     /*iw*/
    }       /*ih*/
  }         /*oc8*/

  PostConvFuncFp16C8(tmp, dst, bias, output_channel, output_plane, conv_param->output_channel_, conv_param->act_type_);
  return NNACL_OK;
}
