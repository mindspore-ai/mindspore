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

#ifndef MINDSPORE_LITE_NNACL_FP32_COMMON_FUNC_H_
#define MINDSPORE_LITE_NNACL_FP32_COMMON_FUNC_H_

#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/conv_parameter.h"

typedef struct ConvDwFp32BorderParam {
  float *dst;
  const float *src;
  const float *weight;
  const float *bias;
  size_t height;
  size_t width;
  size_t in_kh_step;
  size_t in_kw_step;
  size_t kernel_w;
  size_t relu;
  size_t relu6;
} ConvDwFp32BorderParam;

#ifdef __cplusplus
extern "C" {
#endif

void PostConvFuncFp32C8(const float *c8_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t stride, size_t relu_type);
void PostConvFuncFp32C4(const float *c4_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t plane_stride, size_t relu_type);

void WinogradTransLeft(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length);
void WinogradTransRight(const float *S, const float *B, float *M, size_t w, size_t h, size_t k, size_t length);

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
void ConvDwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step,
                      size_t in_sw_step, size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
#ifdef ENABLE_AVX
void ConvDwFp32Border(ConvDwFp32BorderParam *param);
#else
void ConvDwFp32Border(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
                      size_t in_kh_step, size_t in_kw_step, size_t kernel_w, size_t relu, size_t relu6);
#endif
void DeconvDwFp32Center(float *dst, const float *src, const float *weight, size_t height, size_t width, size_t kernel_h,
                        size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
                        size_t in_kh_step, size_t in_kw_step);
void PostFuncBiasReluC8(float *dst, const float *src, const float *bias, size_t oc8div, size_t oc8mod,
                        size_t plane_size, size_t stride, size_t relu_type);
void ConvDwFp32Row(float *output_ptr, const float *input_ptr, const float *weight_ptr, size_t num_pixels,
                   size_t output_channel, size_t input_step);
void PostFuncBiasReluC4(float *dst, const float *src, const float *bias, size_t oc4div, size_t oc4mod,
                        size_t plane_size, size_t plane_stride, size_t relu_type);
#endif

#ifdef ENABLE_ARM64
void BiasAdd(const float *bias, float *data, size_t oc4, size_t plan_size);
void BiasAddRelu6(const float *bias, float *data, size_t oc4, size_t plan_size);
void BiasAddRelu(const float *bias, float *data, size_t oc4, size_t plan_size);
void Relu6(float *data, size_t element4);
void Relu(float *data, size_t element4);

void DeconvDwFp32Border(float *dst, const float *src, const float *weight, size_t height, size_t width,
                        size_t in_kh_step, size_t in_kw_step, size_t kernel_w);

void ConvSwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t ic4,
                      size_t in_sh_step, size_t in_sw_step, size_t in_kh_step, size_t in_kw_step, size_t relu,
                      size_t relu6);

void ConvDw3x3Stride1(float *output, const float *buffer, const float *weight, const float *bias, int col_size,
                      int row_size, int channel, int output_h, int output_w, size_t relu, size_t relu6);

void ConvDw3x3Stride2(float *output, const float *buffer, const float *weight, const float *bias, int col_size,
                      int row_size, int channel, int output_h, int output_w, size_t relu, size_t relu6);

void ConvDw3x3Corner(float *dst, const float *src, const float *weight, const float *bias, int in_kh_step,
                     int in_kw_step, int channel, size_t relu, size_t relu6);

void ConvDw3x3Vertical(float *dst, const float *src, const float *weight, const float *bias, int in_kh_step,
                       int in_kw_step, int channel, size_t relu, size_t relu6);

void ConvDw3x3Horizontal(float *dst, const float *src, const float *weight, const float *bias, int in_kh_step,
                         int in_kw_step, int channel, size_t relu, size_t relu6);
#endif

#ifdef __cplusplus
}
#endif
#endif /* MINDSPORE_LITE_NNACL_FP32_COMMON_FUNC_H_ */
