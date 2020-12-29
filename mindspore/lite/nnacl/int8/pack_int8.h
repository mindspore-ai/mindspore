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

#ifndef MINDSPORE_LITE_NNACL_INT8_PACK_INT8_H_
#define MINDSPORE_LITE_NNACL_INT8_PACK_INT8_H_

#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/conv_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void PackNHWCToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWC4ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNHWC8Int8(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWC8ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel);
void PackNCHWToNC8HW8Int8(const void *src, void *dst, int batch, int plane, int channel);
void PackNC4HW4ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToC8HWN8Int8(const void *src, void *dst, int batch, int plane, int channel);
void PackNCHWToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel);
void PackNHWCToNCHWInt8(const void *src, void *dst, int batch, int plane, int channel);

void PackInputSum16x4Int8(const int8_t *input, int32_t *input_sum, int32_t *filter_zp, ConvParameter *conv_param);
void PackInputSum16x4PerLayer(const int8_t *src, int32_t *dst, int32_t filter_zp, size_t row4, size_t col16);
void PackInputToC8Int8(const int8_t *input_data, int16_t *packed_input, ConvParameter *conv_param);
void PackWeightToC8Int8(const int8_t *origin_weight_data, int16_t *packed_weight_data, ConvParameter *conv_param);
void Im2ColPackUnitInt8Opt(const int8_t *input_data, int8_t *packed_input, int8_t *matmul_input, int real_cal_num,
                           int block_index, int32_t *filter_zp, int32_t *input_sum, ConvParameter *conv_param,
                           bool per_channel, bool is_optimize);
#ifdef ENABLE_ARM
void PreSum4x16Int8Pert(const int8_t *src, int32_t *sum, size_t row4, size_t col16, int32_t filter_zp);
void PreSum4x16Int8Peroc(const int8_t *src, int32_t *sum, int32_t *zp, size_t hw4, size_t ic16, int32_t oc_div,
                         size_t oc_res, size_t stride);
#endif

void PackDepthwiseInt8Input(const int8_t *src, int16_t *dst, const ConvParameter *conv_param);
void PackDepthwiseInt8Weight(const int8_t *origin_weight, int16_t *packed_weight_, int plane, int channel,
                             ConvQuantArg *quant_qrg);
void PackDeconvDepthwiseInt8Weight(const int8_t *origin_weight, int16_t *packed_weight_, int plane, int channel,
                                   ConvQuantArg *quant_qrg);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_INT8_PAD_INT8_H_
