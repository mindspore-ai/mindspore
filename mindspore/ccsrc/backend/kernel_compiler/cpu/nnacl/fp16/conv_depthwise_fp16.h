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

#ifndef MINDSPORE_NNACL_FP16_CONV_DEPTHWISE_FP16_H_
#define MINDSPORE_NNACL_FP16_CONV_DEPTHWISE_FP16_H_

#include "nnacl/conv_parameter.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif
void ConvDwFp16Row(float16_t *output_ptr, const float16_t *input_ptr, const float16_t *filter_ptr, size_t num_pixels,
                   size_t input_channel, size_t input_step);
#ifdef ENABLE_ARM64
void ConvDwFp16Border(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias,
                      size_t height, size_t width, size_t in_kh_step, size_t in_kw_step, size_t kernel_w, size_t relu,
                      size_t relu6);
void ConvDwFp16Center(float16_t *dst, const float16_t *src, const float16_t *weight, const float16_t *bias,
                      size_t height, size_t width, size_t kernel_h, size_t kernel_w, size_t out_h_step,
                      size_t block_channel, size_t in_sh_step, size_t in_sw_step, size_t in_kh_step, size_t in_kw_step,
                      size_t relu, size_t relu6);
void DeconvDwFp16Border(float16_t *dst, const float16_t *src, const float16_t *weight, size_t height, size_t width,
                        size_t in_kh_step, size_t in_kw_step, size_t kernel_w);
void DeconvDwFp16Center(float16_t *dst, const float16_t *src, const float16_t *weight, size_t height, size_t width,
                        size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step,
                        size_t in_sw_step, size_t in_kh_step, size_t in_kw_step);
#endif

void ConvDwFp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                const float16_t *bias_data, const ConvParameter *conv_param, int task_id);

void ConvDwC8Fp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                  const float16_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                  int task_id);

void DeconvDwC8Fp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                    const float16_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                    int task_id);

#ifdef ENABLE_ARM
void ConvDw3x3LineFp16(float16_t *dst, float16_t **lines, const float16_t *weight, const float16_t *bias_data,
                       int width, int ori_channel, bool relu, bool relu6);
void ConvDw3x3Fp16(float16_t *output_data, float16_t *buffer, const float16_t *input_data, const float16_t *weight_data,
                   const float16_t *bias_data, const ConvParameter *conv_param, int start_oh, int end_oh);
#endif

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_CONV_DEPTHWISE_FP16_H_
