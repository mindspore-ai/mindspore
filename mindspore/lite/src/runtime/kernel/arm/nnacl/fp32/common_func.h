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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_COMMON_FUNC_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_COMMON_FUNC_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void PostConvFuncFp32C4(const float *c4_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t stride, bool is_relu, bool is_relu6);
void PostConvFuncFp32C8(const float *c8_out_ptr, float *out_ptr, const float *bias_ptr, size_t output_channel,
                        size_t plane_size, size_t stride, bool is_relu, bool is_relu6);
void MatrixAdd(const float *a_ptr, const float *b_ptr, float *dst, size_t a_stride, size_t b_stride, size_t c_stride,
               size_t row, size_t col);
void MatrixSub(const float *a_ptr, const float *b_ptr, float *dst, size_t a_stride, size_t b_stride, size_t c_stride,
               size_t row, size_t col);
void MatrixMultiAdd(float *c11, float *c12, float *c21, float *c22, float *x_ptr, size_t row, size_t col,
                    size_t c_stride, size_t x_stride);

#ifdef ENABLE_ARM
void ConvDwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step,
                      size_t in_sw_step, size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6);
void DeconvDwFp32Center(float *dst, const float *src, const float *weight, size_t height, size_t width, size_t kernel_h,
                        size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
                        size_t in_kh_step, size_t in_kw_step);
#endif

#ifdef ENABLE_ARM64
void BiasAdd(const float *bias, float *data, size_t oc4, size_t plan_size);
void BiasAddRelu6(const float *bias, float *data, size_t oc4, size_t plan_size);
void BiasAddRelu(const float *bias, float *data, size_t oc4, size_t plan_size);
void Relu6(float *data, size_t element4);
void Relu(float *data, size_t element4);
void C4BiasAdd(float *dst, const float *input, const float *bias, size_t oc, size_t plane_size, size_t stride);
void C4BiasAddRelu(float *dst, const float *input, const float *bias, size_t oc, size_t plane_size, size_t stride);
void C4BiasAddRelu6(float *dst, const float *input, const float *bias, size_t oc, size_t plane_size, size_t stride);
void C4Relu(float *dst, const float *input, size_t oc, size_t plane_size, size_t stride);
void C4Relu6(float *dst, const float *input, size_t oc, size_t plane_size, size_t stride);

void ConvDwFp32Border(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
                      size_t in_kh_step, size_t in_kw_step, size_t kernel_w, size_t relu, size_t relu6);
void PostFuncBiasReluC8(float *dst, const float *src, const float *bias, size_t oc8div, size_t oc8mod,
                        size_t plane_size, size_t stride, size_t relu_type);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_COMMON_FUNC_H_ */
