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
#include <float.h>
int DeConvPostFp16(const float16_t *src, float16_t *tmp, const float16_t *bias, float16_t *dst, int output_channel,
                   const ConvParameter *conv_param) {
  float16x8_t min_v = vdupq_n_f16(-FLT_MAX);
  float16x8_t max_v = vdupq_n_f16(FLT_MAX);
  if (conv_param->act_type_ == ActType_Relu) {
    min_v = vdupq_n_f16(0.f);
  }
  if (conv_param->act_type_ == ActType_Relu6) {
    min_v = vdupq_n_f16(0.f);
    max_v = vdupq_n_f16(6.f);
  }

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

  NNACL_CHECK_ZERO_RETURN_ERR(conv_param->dilation_h_);
  NNACL_CHECK_ZERO_RETURN_ERR(conv_param->dilation_w_);

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

        const float16_t *src_in_ptr = src_ptr + ih * src_ih_stride + iw * src_iw_stride;
        float16_t *dst_in_ptr = dst_ptr + oh * dst_oh_stride + ow * dst_ow_stride;

        for (int kh = kh_start; kh < kh_end; kh++) {
          const float16_t *src_kh_ptr = src_in_ptr + kh * src_kh_stride;
          float16_t *dst_kh_ptr = dst_in_ptr + kh * dst_kh_stride;

          for (int kw = kw_start; kw < kw_end; kw++) {
            const float16_t *src_kw_index = src_kh_ptr + kw * src_kw_stride;
            float16_t *dst_kw_index = dst_kh_ptr + kw * dst_kw_stride;
#ifdef ENABLE_ARM64
            asm volatile(
              "mov x0, %[src_kw_index] \n"
              "mov x1, %[dst_kw_index] \n"

              "ld1 {v0.8h}, [x0] \n"
              "ld1 {v1.8h}, [x1] \n"

              "fadd v0.8h, v0.8h, v1.8h \n"

              "st1 {v0.8h}, [x1] \n"

              :
              : [ src_kw_index ] "r"(src_kw_index), [ dst_kw_index ] "r"(dst_kw_index)
              : "x0", "x1", "v0", "v1");
#else
            for (int i = 0; i < C8NUM; i++) {
              dst_kw_index[i] += src_kw_index[i];
            }
#endif
          }  // kw
        }    // kh
      }      // iw
    }        // ih

    /* add bias for current oh*ow*C8
     * write to output data ptr in nhwc format */
    float16x8_t bias_v = vld1q_f16(bias + c);
    float16_t *pack_tmp_data = dst_ptr;
    for (size_t i = 0; i < output_plane; i++) {
      float16x8_t data_v = vld1q_f16(pack_tmp_data);
      data_v = vaddq_f16(data_v, bias_v);
      data_v = vminq_f16(data_v, max_v);
      data_v = vmaxq_f16(data_v, min_v);
      vst1q_f16(pack_tmp_data, data_v);
      pack_tmp_data += C8NUM;
    }
  }  // oc8
  return NNACL_OK;
}
