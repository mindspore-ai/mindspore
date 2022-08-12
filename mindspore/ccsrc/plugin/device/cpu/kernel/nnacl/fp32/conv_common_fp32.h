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
#include "nnacl/fp32/conv_sw_avx_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*Row2ColMajorFuncPtr)(const float *src_ptr, float *dst_ptr, int row, int col);
#ifdef ENABLE_ARM64
typedef void (*MatmulFloatOptFuncPtr)(const float *a, const float *b, float *c, const float *bias, int act_type,
                                      int depth, int row, int col, size_t stride, size_t write_mode);
#endif

void Im2ColPackUnitFp32(const float *input_data, const ConvParameter *conv_param, float *packed_input, int real_cal_num,
                        int block_index);

// fp32 convolution common (im2col+gemm)
void ConvFp32(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
              float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param);

// fp32 convolution common (im2col+gemm)
void ConvFp32CutByBatch(const float *input_data, float *packed_input, const float *packed_weight,
                        const float *bias_data, float *col_major_input, float *output_data, int task_id,
                        const ConvParameter *conv_param);

// common convolution output C4HW4, if out_channel mod 4 remains, just output real channel, no zeros padded.
void ConvFp32OutNC4HW4(const float *input_data, float *packed_input, const float *packed_weight, const float *bias_data,
                       float *col_major_input, float *output_data, int task_id, const ConvParameter *conv_param);

#ifdef ENABLE_AVX
void CommonConv6x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t depth,
                          size_t out_step, size_t act_flag, size_t real_cal_row);
#endif

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CONV_COMMON_H_
