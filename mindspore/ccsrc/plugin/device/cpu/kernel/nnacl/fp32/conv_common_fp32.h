/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_FP32_CONV_COMMON_H_
#define MINDSPORE_NNACL_FP32_CONV_COMMON_H_

#include "nnacl/pack.h"
#include "nnacl/op_base.h"
#include "nnacl/common_func.h"
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*Row2ColMajorFuncPtr)(const float *src_ptr, float *dst_ptr, int row, int col);
#ifdef ENABLE_ARM64
typedef void (*MatmulFloatOptFuncPtr)(const float *a, const float *b, float *c, const float *bias, int act_type,
                                      int depth, int row, int col, size_t stride, size_t write_mode);
#endif

// fp32 convolution common (im2col+gemm)
void ConvFp32(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
              float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param);

// common convolution output C4HW4, if out_channel mod 4 remains, just output real channel, no zeros padded.
void ConvFp32OutNC4HW4(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
                       float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param);

#ifdef ENABLE_AVX
void CommonConv6x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t depth,
                          size_t out_step, size_t act_flag, size_t real_cal_row);

typedef void (*SWConvKernel)(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                             size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                             size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step,
                             size_t kw_remainder, size_t write_mode);

void SWBorder(float *dst, const float *src, const float *weight, const float *bias, int top, int bottom, int left,
              int right, const ConvParameter *conv_param, const SlidingWindowParam *sw_param, SWConvKernel kernel,
              int act_type, int ow_bock, int oc_block, size_t write_mode);

void ConvSWFp32(const float *input_data, const float *packed_weight, const float *bias_data, float *output_data,
                int task_id, ConvParameter *conv_param, SlidingWindowParam *sw_param);
void SWCenter(float *dst, const float *src, const float *weight, const float *bias, int height, int width, int kernel_h,
              int kernel_w, bool is_relu, bool is_relu6, SlidingWindowParam *sw_param);

#ifdef ENABLE_DEBUG
void SWConvWxKKernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                     size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                     size_t write_mode);
#endif

void SWConv3x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                      size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                      size_t write_mode);

void SWConv1x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                      size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                      size_t write_mode);

void SWConv4x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                      size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                      size_t write_mode);

void SWConv1x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                      size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                      size_t write_mode);

void SWConv6x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                      size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                      size_t write_mode);

void SWConv1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                      size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                      size_t write_mode);

void SWConv12x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                      size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                      size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                      size_t write_mode);

void SWConv8x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                     size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                     size_t write_mode);

void SWConv4x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                     size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                     size_t write_mode);

void SWConv1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                     size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                     size_t ic_algin, size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder,
                     size_t write_mode);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CONV_COMMON_H_
