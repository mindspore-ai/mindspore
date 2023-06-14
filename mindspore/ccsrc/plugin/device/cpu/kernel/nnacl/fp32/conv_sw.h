/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_CONV_SW_H_
#define MINDSPORE_NNACL_FP32_CONV_SW_H_

#define GenerateConvSWFunc(backend, oc_unit_num, row_num_list, kernel_list, compute_core, outer_compute)            \
  void SWBorder##backend(float *dst, const float *src, const float *weight, const float *bias, int top, int bottom, \
                         int left, int right, const ConvParameter *conv_param, const SlidingWindowParam *sw_param,  \
                         const SWConvKernel kernel, int act_type, int ow_bock, int oc_block, size_t write_mode) {   \
    for (int oh = top; oh < bottom; oh++) {                                                                         \
      int ih = oh * conv_param->stride_h_ - conv_param->pad_u_;                                                     \
      int start_kh = MSMAX(0, UP_DIV(-ih, conv_param->dilation_h_));                                                \
      int end_kh = MSMIN(conv_param->kernel_h_, UP_DIV(conv_param->input_h_ - ih, conv_param->dilation_h_));        \
      const float *src_h = src + ih * sw_param->in_h_step_;                                                         \
      float *dst_kernel = dst + left * sw_param->out_w_step_;                                                       \
      for (int ow = left; ow < right; ow += ow_bock) {                                                              \
        int iw = ow * conv_param->stride_w_ - conv_param->pad_l_;                                                   \
        int start_kw = MSMAX(0, UP_DIV(-iw, conv_param->dilation_w_));                                              \
        int end_kw = MSMIN(conv_param->kernel_w_, UP_DIV(conv_param->input_w_ - iw, conv_param->dilation_w_));      \
        const float *src_w = src_h + iw * sw_param->ic_align_;                                                      \
        const float *src_kernel = src_w + start_kh * sw_param->in_kh_step_ + start_kw * sw_param->in_kw_step_;      \
        const float *weight_kernel =                                                                                \
          weight + (start_kh * conv_param->kernel_w_ + start_kw) * sw_param->ic_align_ * C8NUM * oc_block;          \
        outer_compute dst_kernel += ow_bock * sw_param->out_w_step_;                                                \
      }                                                                                                             \
      dst += sw_param->out_h_step_;                                                                                 \
    }                                                                                                               \
  }                                                                                                                 \
                                                                                                                    \
  void ConvSW##backend##Fp32(const float *input_data, const float *packed_weight, const float *bias_data,           \
                             float *output_data, int task_id, ConvParameter *conv_param,                            \
                             SlidingWindowParam *sw_param) {                                                        \
    int out_h = conv_param->output_h_;                                                                              \
    int oh_step = UP_DIV(out_h, conv_param->thread_num_);                                                           \
    int oh_start = oh_step * task_id;                                                                               \
    int oh_end = MSMIN(oh_start + oh_step, out_h);                                                                  \
    if (oh_start >= oh_end) {                                                                                       \
      return;                                                                                                       \
    }                                                                                                               \
    int oc_tile_ = C8NUM; /* oc in algin to C8NUM in arm64 */                                                       \
    int act_type = 0;                                                                                               \
    if (conv_param->act_type_ == ActType_Relu6) {                                                                   \
      act_type += 1;                                                                                                \
    }                                                                                                               \
    if (conv_param->act_type_ == ActType_Relu || conv_param->act_type_ == ActType_Relu6) {                          \
      act_type += 2;                                                                                                \
    }                                                                                                               \
    int kernel_h = conv_param->kernel_h_;                                                                           \
    int kernel_w = conv_param->kernel_w_;                                                                           \
    int ic_algin = sw_param->ic_align_;                                                                             \
    int in_sw_step = sw_param->in_sw_step_;                                                                         \
    int in_kw_step = sw_param->in_kw_step_;                                                                         \
    int in_kh_step = sw_param->in_kh_step_;                                                                         \
    int in_sh_step = sw_param->in_sh_step_;                                                                         \
    int out_h_step = sw_param->out_h_step_;                                                                         \
    int out_c_step = sw_param->out_c_step_;                                                                         \
    int out_w_step = sw_param->out_w_step_;                                                                         \
    int out_block_step = sw_param->out_block_step_;                                                                 \
    int kernel_step = sw_param->kernel_step_;                                                                       \
    int in_step = sw_param->in_step_;                                                                               \
    int out_step = sw_param->out_step_;                                                                             \
    int c_block = sw_param->c_block_;                                                                               \
    int top = sw_param->top_;                                                                                       \
    int left = sw_param->left_;                                                                                     \
    int right = sw_param->right_;                                                                                   \
    int bottom = sw_param->bottom_;                                                                                 \
    int stride_h = conv_param->stride_h_;                                                                           \
    int stride_w = conv_param->stride_w_;                                                                           \
    int out_w = conv_param->output_w_;                                                                              \
    int pad_u = conv_param->pad_u_;                                                                                 \
    int pad_l = conv_param->pad_l_;                                                                                 \
    int in_h_step = sw_param->in_h_step_;                                                                           \
    int out_batch = conv_param->output_batch_;                                                                      \
    int in_h_start = top * stride_h - pad_u;                                                                        \
    int in_w_start = left * stride_w - pad_l;                                                                       \
    int center_step = in_h_start * in_h_step + in_w_start * ic_algin;                                               \
    int write_mode = conv_param->out_format_;                                                                       \
    row_num_list kernel_list for (int b = 0; b < out_batch; b++) {                                                  \
      for (int oh = oh_start; oh < oh_end; oh += 1) {                                                               \
        float *dst_oh = output_data + oh * out_h_step;                                                              \
        const float *src_h = input_data + center_step;                                                              \
                                                                                                                    \
        int oc_block = 0;                                                                                           \
        const float *bias = bias_data;                                                                              \
        for (int oc = 0; oc < c_block; oc += oc_block) {                                                            \
          oc_block = MSMIN(oc_unit_num, c_block - oc);                                                              \
          const float *weight = packed_weight + oc * kernel_step;                                                   \
          if (bias != NULL) {                                                                                       \
            bias = bias_data + oc * oc_tile_;                                                                       \
          }                                                                                                         \
          /* nhwc dst_w = dst_oh + oc * oc_tile_;  nc8hw8 dst_w = dst_oh * oc * ow * oh * oc_tile_; */              \
          float *dst_oc = dst_oh + oc * out_c_step;                                                                 \
          const SWConvKernel kernel_border = kernel[oc_block - 1][0];                                               \
          if (oh < top || oh >= bottom) { /* oh in up or down border */                                             \
            SWBorder##backend(dst_oc, input_data, weight, bias, oh, oh + 1, 0, out_w, conv_param, sw_param,         \
                              kernel_border, act_type, 1, oc_block, write_mode);                                    \
          } else { /* oh in center */                                                                               \
            /* ow in right */                                                                                       \
            SWBorder##backend(dst_oc, input_data, weight, bias, oh, oh + 1, 0, left, conv_param, sw_param,          \
                              kernel_border, act_type, 1, oc_block, write_mode);                                    \
            /* ow in center */                                                                                      \
            const float *src_w = src_h + (oh - top) * in_sh_step;                                                   \
            int ow_block = ow_block_num[oc_block - 1];                                                              \
            for (int ow = left; ow < right; ow += ow_block) { /* left ~ right */                                    \
              ow_block = MSMIN(ow_block, right - ow);                                                               \
              compute_core src_w += ow_block * in_sw_step;                                                          \
            }                                                                                                       \
            /* ow in left */                                                                                        \
            SWBorder##backend(dst_oc, input_data, weight, bias, oh, oh + 1, right, out_w, conv_param, sw_param,     \
                              kernel_border, act_type, 1, oc_block, write_mode);                                    \
          }                                                                                                         \
        }                                                                                                           \
      } /* output h loop */                                                                                         \
      input_data += in_step;                                                                                        \
      output_data += out_step;                                                                                      \
    } /* batch loop */                                                                                              \
  }
#endif  // MINDSPORE_NNACL_FP32_CONV_SW_H_
