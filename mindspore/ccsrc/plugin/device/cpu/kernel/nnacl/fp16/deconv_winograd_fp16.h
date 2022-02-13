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

#ifndef MINDSPORE_NNACL_FP16_DECONV_WINOGRAD_FP16_H_
#define MINDSPORE_NNACL_FP16_DECONV_WINOGRAD_FP16_H_

#include "nnacl/fp16/winograd_transform_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

int PackDeConvWgDataFp16(const float16_t *nhwc_weight, DeConvComputeUnit *unit, const ConvParameter *conv_param,
                         const DeConvParam *deconv_param);

void DeconvWgFp16(const float16_t *nhwc_input_, float16_t *tile_in, float16_t *tile_out, int start_index,
                  int calculate_count, const ConvParameter *conv_param, DeConvParam *deconv_param, int task_id);

void DeconvWgPostFp16(const float16_t *tile_out, float16_t *nc4hw4_output, const ConvParameter *conv_param,
                      const DeConvParam *deconv_param, int calculate_count, int tile_index);

void TiledC4MatmulFp16(float16_t *dst, const float16_t *src, const float16_t *weight, size_t ic4, size_t cal_num,
                       size_t oc4);

void WinogradTransLeftFp16(const float16_t *S, const float16_t *B, float16_t *M, size_t w, size_t h, size_t k,
                           size_t length);

void WinogradTransRightFp16(const float16_t *S, const float16_t *B, float16_t *M, size_t w, size_t h, size_t k,
                            size_t length);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_DECONV_WINOGRAD_FP16_H_
