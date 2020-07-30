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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_PACK_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_PACK_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "src/runtime/kernel/arm/opclib/conv_parameter.h"
#include "src/runtime/kernel/arm/opclib/op_base.h"

#ifdef ENABLE_FP16
void Im2ColPackUnitFp16(float16_t *input_data, ConvParameter *conv_param, float16_t *packed_input, int real_cal_num,
                        int block_index);

void PackWeightFp16(float16_t *weight_data, ConvParameter *conv_param, float16_t *packed_weight);

void PackWeightToC8Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data, ConvParameter *conv_param);

void PackWeightToC4Fp16(const float16_t *origin_weight_data, float16_t *packed_weight_data, ConvParameter *conv_param);

void PackNHWCToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNC4HW4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWC4Fp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNCHWFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNC8HW8ToNHWCFp16(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWFp32ToNC8HW8Fp16(float *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWCFp32ToNHWC8Fp16(float *src, float16_t *dst, int batch, int plane, int channel);

void PackNHWC8Fp16ToNHWCFp32(float16_t *src, float *dst, int batch, int plane, int channel);
#endif
void Im2ColPackUnitFp32(const float *input_data, ConvParameter *conv_param, float *packed_input, int real_cal_num,
                        int block_index);

void Im2ColPackUnitInt8(const int8_t *input_data, int8_t *packed_input, int real_cal_num, int block_index,
                        int32_t *input_sum, ConvParameter *conv_param);

void Im2ColPackUnitInt8Opt(const int8_t *input_data, int8_t *packed_input, int real_cal_num, int block_index,
                           int32_t *input_sum, ConvParameter *conv_param);

void Conv1x1InputPackFp32(const float *src, float *dst, ConvParameter *conv_param);

void Pack1x1WeightFp32(const float *weight_data, float *packed_weight, ConvParameter *conv_param);

void MatrixPack(const float *src, float *dst, int row, int ic4, int stride);

void PackInputToC8Int8(const int8_t *input_data, int16_t *packed_input, ConvParameter *conv_param);

void PackWeightFp32(float *weight_data, ConvParameter *conv_param, float *packed_weight);

void PackWeightInt8(int8_t *weight_data, ConvParameter *conv_param, int8_t *packed_weight, int32_t *weight_sum);

void PackWeightInt8Opt(int8_t *weight_data, ConvParameter *conv_param, int8_t *packed_weight, int32_t *weight_sum);

void PackWeightToC8Int8(const int8_t *origin_weight_data, int16_t *packed_weight_data, ConvParameter *conv_param);

void PackNHWCToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNC4HW4Fp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWC4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWC4Fp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWCFp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNCHWFp32(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWC4ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWC4Int8(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel);

void PackNC4HW4ToNCHWInt8(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToC8HWN8Int8(const void *src, void *dst, int batch, int plane, int channel);

void PackNHWCToNC8HW8Int8(const void *src, void *dst, int batch, int plane, int channel);

void PackNCHWToNHWCInt8(const void *src, void *dst, int batch, int plane, int channel);

void PackDepthwiseInt8Input(const int8_t *src, int16_t *dst, const ConvParameter *conv_param);

void PackDepthwiseInt8Weight(const int8_t *src, int16_t *dst, const ConvParameter *conv_param);

inline void UnpackHwcToChwFp32(float *src_ptr, float *dst_ptr, int channel, int h, int w) {
  int cur = 0;
  for (int i = 0; i < channel; i++) {
    auto plane = i / BLOCK;
    auto offset = i % BLOCK;
    auto src_plane = plane * h * w * BLOCK + src_ptr;
    for (int j = 0; j < h * w; j++) {
      dst_ptr[cur++] = src_plane[j * BLOCK + offset];
    }
  }
}

inline void C8UnpackToHwcFp32(float *src_ptr, float *dst_ptr, int channel, int h, int w) {
  int cur = 0;
  for (int j = 0; j < h * w; j++) {
    for (int i = 0; i < channel; i++) {
      auto plane = i / 8;
      auto offset = i % 8;
      auto src_plane = plane * h * w * 8 + src_ptr;
      dst_ptr[cur++] = src_plane[j * 8 + offset];
    }
  }
}

inline void C4UnpackToHwcFp32(float *src_ptr, float *dst_ptr, int channel, int h, int w) {
  int cur = 0;
  for (int j = 0; j < h * w; j++) {
    for (int i = 0; i < channel; i++) {
      auto plane = i / 4;
      auto offset = i % 4;
      auto src_plane = plane * h * w * 4 + src_ptr;
      dst_ptr[cur++] = src_plane[j * 4 + offset];
    }
  }
}

inline void C4UnpackToHwcInt8(int8_t *src_ptr, int8_t *dst_ptr, int channel, int h, int w) {
  int cur = 0;
  for (int j = 0; j < h * w; j++) {
    for (int i = 0; i < channel; i++) {
      auto plane = i / 4;
      auto offset = i % 4;
      auto src_plane = plane * h * w * 4 + src_ptr;
      dst_ptr[cur++] = src_plane[j * 4 + offset];
    }
  }
}

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_PACK_H_
