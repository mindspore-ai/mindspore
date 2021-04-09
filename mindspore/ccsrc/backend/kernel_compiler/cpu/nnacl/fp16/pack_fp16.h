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

#ifndef MINDSPORE_NNACL_FP16_PACK_FP16_H_
#define MINDSPORE_NNACL_FP16_PACK_FP16_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void Im2ColPackUnitFp16(float16_t *input_data, ConvParameter *conv_param, float16_t *packed_input, int real_cal_num,
                        int block_index);

void PackWeightToC8Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data, ConvParameter *conv_param);

void PackHWCToWHCFp16(const float16_t *src, float16_t *dst, int height, int width, int channel);

void PackWeightToC4Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data, ConvParameter *conv_param);

void PackNHWCToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNCHWFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNHWC8Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWC4ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNCHWFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC8HW8ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWFp32ToNC8HW8Fp16(float *src, float16_t *dst, int batch, int plane, int channel);

void PackNCHWFp16ToNC8HW8Fp16(float16_t *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWCFp32ToNHWC8Fp16(float *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWCFp32ToC8HWN8Fp16(float *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWCFp16ToC8HWN8Fp16(float16_t *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWC8Fp16ToNHWCFp32(float16_t *src, float *dst, int batch, int plane, int channel);

void PackNHWC8ToNHWCFp16(float16_t *src, float16_t *dst, int batch, int plane, int channel);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_PACK_FP16_H_
