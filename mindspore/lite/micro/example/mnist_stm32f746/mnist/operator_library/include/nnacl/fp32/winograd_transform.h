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

#ifndef MINDSPORE_LITE_NNACL_WINOGRAD_TRANSFORM_H_
#define MINDSPORE_LITE_NNACL_WINOGRAD_TRANSFORM_H_

#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif
#include <string.h>
#include "nnacl/pack.h"
#include "nnacl/fp32/winograd_utils.h"

#ifdef __cplusplus
extern "C" {
#endif
// for fp32 winograd input/output transform
void WinogradInputTransform(const float *input_data, float *trans_input, float *tmp_data, int cal_num,
                            int out_tile_index, int out_w_block_num, const ConvParameter *conv_param,
                            InputTransFunc func);

void WinogradOutputTransform(const float *gemm_out, float *out_data, const float *bias_data, int cal_num,
                             int out_tile_index, int output_unit_num, const ConvParameter *conv_param,
                             OutputTransFunc func);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_WINOGRAD_TRANSFORM_H_
