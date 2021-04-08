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

#ifndef MINDSPORE_NNACL_FP16_WINOGRAD_TRANSFORM_FP16_H_
#define MINDSPORE_NNACL_FP16_WINOGRAD_TRANSFORM_FP16_H_

#include <arm_neon.h>
#include <string.h>
#include "nnacl/errorcode.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/conv_fp16.h"
#include "nnacl/fp16/matrix_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

// for fp16 convolution 3x3 filter/input/output transform
void Conv3x3Fp16InputUnit(float16_t *tmp_data, float16_t *trans_input_data, size_t step);

void Conv3x3Fp16InputTransform(const float16_t *input_data, float16_t *trans_input, float16_t *tmp_data,
                               int start_index, int real_cal_num, int out_w_block, ConvParameter *conv_param);

void Conv3x3Fp16FilterTransform(const float16_t *weight_data, float16_t *trans_weight, int iC8, int output_channel,
                                int kernel_plane);

void Conv3x3Fp16OutputUnit(const float16_t *gemm_out, const float16_t *bias_data, float16_t *output_data, int output_w);

void Conv3x3Fp16OutputTransform(const float16_t *gemm_out, float16_t *out_data, const float16_t *bias_data,
                                int start_index, int real_cal_num, int out_w_block, ConvParameter *conv_param);

// fp16 common winograd
void WinogradInputTransformFp16(const float16_t *input_data, float16_t *trans_input, float16_t *tmp_data, int cal_num,
                                int out_tile_index, int out_w_block_num, ConvParameter *conv_param,
                                InputTransFp16Func func);

void WinogradOutputTransformFp16(const float16_t *gemm_out, float16_t *tmp_out_data, const float16_t *bias_data,
                                 int cal_num, int out_tile_index, int output_unit_num, ConvParameter *conv_param,
                                 OutputTransFp16Func func);

// fp16 winograd weight trans
int WinogradWeightTransformFp16(const float16_t *weight_data, float16_t *winograd_data, float *matrix_g,
                                float *matrix_gt, int oc_block, int input_unit, int kernel_unit, int filter_channel,
                                int filter_batch, bool pack);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_WINOGRAD_TRANSFORM_FP16_H_
