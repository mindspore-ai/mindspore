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

#ifndef MINDSPORE_LITE_NNACL_INT8_COMMON_FUNC_H_
#define MINDSPORE_LITE_NNACL_INT8_COMMON_FUNC_H_

#include <string.h>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void PostFuncInt8C4(const int32_t *in, const int32_t *bias, int8_t *out, size_t oc, size_t plane, size_t stride,
                    int32_t multiplier, int32_t left_shift, int32_t right_shift, int32_t zp, int32_t mini,
                    int32_t maxi);
#ifdef ENABLE_ARM
void ConvDwInt8Row(int32_t *output_ptr, const int8_t *input_ptr, const int16_t *weight_ptr, int num_pixels,
                   int output_channel, int input_step, int8_t input_zp);
void ConvDwInt8PostAlign4PerChannel(int8_t *dst, int32_t *buffer, int channel4, int32_t output_zp,
                                    int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift, int32_t acc_min,
                                    int32_t acc_max);
void ConvDwInt8PostAlign4(int8_t *dst, int32_t *buffer, int num_pixels, int32_t output_zp, int32_t out_multiplier,
                          int32_t left_shift, int32_t right_shift, int32_t acc_min, int32_t acc_max);
void IndirectGemmInt16to32_8x4(int32_t *dst, const int16_t *src, const int16_t *weight, size_t ksize, size_t ic8,
                               size_t oc4, size_t offset);
void ConvDwInt8Center(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, size_t height,
                      size_t width, size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel,
                      size_t in_sh_step, size_t in_sw_step, size_t in_kh_step, size_t in_kw_step, int8_t *in_zp,
                      int32_t *out_zp, int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift,
                      int32_t *acc_min, int32_t *acc_max);
void DeconvDwInt8Center(int32_t *dst, const int16_t *src, const int16_t *weight, size_t height, size_t width,
                        size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step,
                        size_t in_sw_step, size_t in_kh_step, size_t in_kw_step);
void DeconvDwInt8Post(int8_t *dst, int32_t *output_buffer, const int32_t *bias, int block_channel, int pixel_nums,
                      int out_multiplier, int left_shift, int right_shift, int32_t out_zp, int32_t acc_min,
                      int32_t acc_max);
int16x8_t LoadAndAddOffset(int8_t *data, int index, int offset);
int32x4_t ClacScaledInput(int32x4_t input, int32x4_t left_shift_result_vec, int32x4_t input_multiplier_vec,
                          int32x4_t right_shift_vec);
#endif

#ifdef ENABLE_ARM32
void ConvDw3x3Int8BorderPixel(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, int height,
                              int width, int in_kh_step, int in_kw_step, int channel, int8_t in_zp, int32_t out_zp,
                              int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift, int32_t acc_min,
                              int32_t acc_max, size_t per_channel);
#endif

#ifdef ENABLE_ARM64
void PostFuncInt8C4Neon64(const int32_t *in, const int32_t *bias, int8_t *out, size_t oc4div, size_t oc4res,
                          size_t plane, size_t stride, int32_t multiplier, int32_t left_shift, int32_t right_shift,
                          int32_t zp, int32_t mini, int32_t maxi);
void ConvDw3x3Int8Neon64(int8_t *output, const int8_t *input, const int16_t *weight, const int32_t *bias,
                         int input_col_size, int input_row_size, int channel, int output_h, int output_w, int8_t in_zp,
                         int32_t out_zp, int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift,
                         int32_t acc_min, int32_t acc_max, size_t per_channel);
void ConvDw3x3Int8Stride2(int8_t *output, const int8_t *input, const int16_t *weight, const int32_t *bias,
                          int input_col_size, int input_row_size, int channel, int output_h, int output_w, int8_t in_zp,
                          int32_t out_zp, int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift,
                          int32_t acc_min, int32_t acc_max, size_t per_channel);
void ConvDw3x3Int8Corner(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias, size_t in_kh_step,
                         size_t in_kw_step, size_t channel, size_t in_zp, size_t out_zp, int32_t *out_multiplier,
                         int32_t *left_shift, int32_t *right_shift, size_t acc_min, size_t acc_max, size_t per_channel);
void ConvDw3x3Int8Vertical(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias,
                           size_t in_kh_step, size_t in_kw_step, size_t channel, size_t in_zp, size_t out_zp,
                           int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift, size_t acc_min,
                           size_t acc_max, size_t per_channel);
void ConvDw3x3Int8Horizontal(int8_t *dst, const int8_t *src, const int16_t *weight, const int32_t *bias,
                             size_t in_kh_step, size_t in_kw_step, size_t channel, size_t in_zp, size_t out_zp,
                             int32_t *out_multiplier, int32_t *left_shift, int32_t *right_shift, size_t acc_min,
                             size_t acc_max, size_t per_channel);
#endif
#ifdef __cplusplus
}
#endif

#endif /* MINDSPORE_LITE_NNACL_FP32_COMMON_FUNC_H_ */
