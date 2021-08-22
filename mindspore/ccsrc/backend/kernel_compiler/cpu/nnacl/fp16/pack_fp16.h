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

#include "nnacl/conv_parameter.h"
#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
void Im2ColPackUnitFp16(const float16_t *input_data, const ConvParameter *conv_param, float16_t *packed_input,
                        int real_cal_num, int block_index);

void PackWeightToC8Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data,
                        const ConvParameter *conv_param);

void PackHWCToWHCFp16(const float16_t *src, float16_t *dst, int height, int width, int channel);

void PackWeightToC4Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data,
                        const ConvParameter *conv_param);

void PackNHWCToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel, int task_id, int thread_count);

void PackNHWCToNCHWFp16(const void *src, void *dst, int batch, int plane, int channel, int task_id, int thread_count);

void PackNHWCToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNHWC8Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWC4ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNCHWFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWFp32ToNC8HW8Fp16(const float *src, float16_t *dst, int batch, int plane, int channel);

void PackNCHWFp16ToNC8HW8Fp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel);

void PackNC8HW8ToNHWCFp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWCFp32ToNHWC8Fp16(const float *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWCFp32ToC8HWN8Fp16(const float *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWCFp16ToC8HWN8Fp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWC8Fp16ToNHWCFp32(const float16_t *src, float *dst, int batch, int plane, int channel);

void PackNHWC8ToNHWCFp16(const float16_t *src, float16_t *dst, int batch, int plane, int channel);

#ifdef ENABLE_ARM82_A32
void Transpose8x8A32Fp16(const float16_t *src, float16_t *dst, size_t src_stride, size_t dst_stride);

void Transpose12x8A32Fp16(const float16_t *src, float16_t *dst, size_t src_stride, size_t dst_stride);
#endif

#ifdef ENABLE_ARM64
void Transpose4x8ARM64Fp16(const float16_t *src, float16_t *dst, size_t src_stride, size_t dst_stride);
void Transpose8x8ARM64Fp16(const float16_t *src, float16_t *dst, size_t src_stride, size_t dst_stride);
void Transpose12x8ARM64Fp16(const float16_t *src_ptr, float16_t *dst_ptr, size_t src_stride, size_t dst_stride);
void Transpose16x8ARM64Fp16(const float16_t *src, float16_t *dst, size_t src_stride, size_t dst_stride);
#endif

#ifdef ENABLE_ARM
void PackWeightConvDw3x3Fp16(const void *src, void *dst, int channel);
#endif

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_PACK_FP16_H_
