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

#ifndef MINDSPORE_LITE_NNACL_FP32_GRAD_PACK_EXT_H_
#define MINDSPORE_LITE_NNACL_FP32_GRAD_PACK_EXT_H_

#include <stddef.h>
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void RollingIm2ColPackUnitFp32(const float *input_data, const ConvParameter *conv_param, float *packed_input,
                               int real_cal_num, int block_index);
void RollingIm2ColPackDwUnitFp32(const float *input_data, const ConvParameter *conv_param, float *packed_input,
                                 int real_cal_num, int block_index);

void rolling_im2col_hwc(const float *in_data, float *data_col, const ConvParameter *conv_param, int rows, int start);
void rolling_im2row_hwc(const float *in_data, float *data_row, const ConvParameter *conv_param, int rows, int start);
void rolling_col2im_hwc(const float *data_col, float *data_im, const ConvParameter *conv_param, int rows, int start);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_GRAD_PACK_EXT_H_
