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

#ifndef MINDSPORE_NNACL_FP32_CONV_DEPTHWISE_H_
#define MINDSPORE_NNACL_FP32_CONV_DEPTHWISE_H_

#include "nnacl/conv_parameter.h"
#include "nnacl/base/conv_common_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ENABLE_ARM64
void DepthwiseCenter(float *dst, const float *src, const float *weight, const float *bias, int height, int width,
                     int kernel_h, int kernel_w, int out_h_step, int block_channel, int in_sh_step, int in_sw_step,
                     int in_kh_step, int in_kw_step, bool is_relu, bool is_relu6);
#endif

int ConvDw(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
           const ConvParameter *conv_param, int task_id);

void InitSlidingParam(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block);

void InitSlidingParamConv(SlidingWindowParam *sliding, const ConvParameter *conv_param, int input_block,
                          int weight_block);

void AppendSlidingParamConv(SlidingWindowParam *sliding, const ConvParameter *conv_param, int in_block,
                            int weight_block);

void InitSlidingParamConvDw(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block);

void AppendSlidingParamConvDw(SlidingWindowParam *sliding, const ConvParameter *conv_param, int block);

void ConvDwSWFp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                  const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id);

bool CheckConvDwUse3X3(const ConvParameter *conv_param);

bool CheckConvDwUseIndirectBuffer(const ConvParameter *conv_param);

void ConvDwInitIndirection(float **indirect_buffer, float *src, float *zero_ptr, const ConvParameter *conv_param,
                           int step_h, int step_w);

#ifdef ENABLE_ARM64
void ConvDwFp32Indirect3x3(float *output, float **input, const float *weights, const float *bias, int channels,
                           int output_width, size_t input_stride, size_t relu, size_t relu6);

void ConvDwFp32Indirect5x5(float *output, float **input, const float *weights, const float *bias, int channels,
                           int output_width, size_t input_stride, size_t relu, size_t relu6);
#endif

#ifdef ENABLE_AVX
typedef void (*DepthwiseSWKernel)(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                                  size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                                  size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW3x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW1x32Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW4x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW1x24Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW4x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                           size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                           size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW8x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                          size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                          size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSW1x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                          size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                          size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);

void DepthwiseSWAvxFp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                        const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id);

void DepthwiseBorderAvxFp32(float *dst, const float *src, const float *weight, const float *bias, int top, int left,
                            int right, const ConvParameter *conv_param, const SlidingWindowParam *sw_param,
                            const DepthwiseSWKernel kernel, int act_type, int ow_bock, int oc_block);

void ConvDwFp32Avx3x3(float *output, float **input, const float *weights, const float *bias, size_t channels,
                      size_t output_width, size_t input_stride, size_t relu, size_t relu6);

void ConvDwFp32Avx5x5(float *output, float **input, const float *weights, const float *bias, size_t channels,
                      size_t output_width, size_t input_stride, size_t relu, size_t relu6);
#ifdef ENABLE_DEBUG
void DepthwiseSWWxKKernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
                          size_t kernel_w, size_t act_flag, size_t ow_block, size_t oc_block, size_t oc_algin,
                          size_t in_kw_step, size_t in_kh_step, size_t in_sw_step, size_t kw_remainder);
#endif
#endif

#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
void ConvDw3x3Line(float *dst, float **lines, const float *weight, const float *bias_data, int width, int ori_channel,
                   bool relu, bool relu6);
void ConvDw3x3(float *output_data, float *buffer, const float *input_data, const float *weight_data,
               const float *bias_data, const ConvParameter *conv_param, int start_oh, int end_oh);
#endif

void ConvDwFp32IndirectRow(float *output, float **input, const float *weights, const float *bias, int channels,
                           int output_width, int input_stride, bool relu, bool relu6, int kernel);

void ConvDwIndirection(float *output_data, float **indirect_buffer, const float *weight_data, const float *bias_data,
                       float *zero_ptr, const ConvParameter *conv_param, int task_id);

void DeconvDwSWFp32(float *output_data, const float *input_data, const float *weight_data, const float *bias_data,
                    const ConvParameter *conv_param, const SlidingWindowParam *sliding, int task_id);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CONV_DEPTHWISE_H_
