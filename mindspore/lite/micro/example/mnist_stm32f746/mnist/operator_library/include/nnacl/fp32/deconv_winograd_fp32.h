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
#ifndef MINDSPORE_LITE_NNACL_FP32_DECONV_WINOGRAD_H_
#define MINDSPORE_LITE_NNACL_FP32_DECONV_WINOGRAD_H_

#include <string.h>
#include "nnacl/pack.h"
#include "nnacl/op_base.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/errorcode.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/base/minimal_filtering_generator.h"

#ifdef __cplusplus
extern "C" {
#endif

int PackDeConvWgDataFp32(const float *nhwc_weight, DeConvComputeUnit *unit, const ConvParameter *conv_param,
                         const DeConvParam *deconv_param);
void DeconvWg(const float *nhwc_input_, float *tile_in, float *tile_out, int start_index, int calculate_count,
              const ConvParameter *conv_param, DeConvParam *deconv_param, int task_id);
void DeconvWgPost(const float *tile_out, float *nc4hw4_output, const ConvParameter *conv_param,
                  const DeConvParam *deconv_param, int calculate_count, int tile_index);
void TiledC4MatmulFp32(float *dst, const float *src, const float *weight, size_t ic4, size_t cal_num, size_t oc4);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_DECONV_WINOGRAD_H_
