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

#ifndef MINDSPORE_NNACL_INT8_QUANTDTYPECAST_H_
#define MINDSPORE_NNACL_INT8_QUANTDTYPECAST_H_

#include "nnacl/op_base.h"

typedef struct QuantDTypeCastParameter {
  OpParameter op_parameter_;
  int32_t srcT;
  int32_t dstT;
  int32_t axis;
} QuantDTypeCastParameter;

#ifdef __cplusplus
extern "C" {
#endif
int DoDequantizeInt8ToFp32(const int8_t *quant_values, float *real_values, float scale, int32_t zp, int size);
int DoQuantizeFp32ToInt8(const float *real_values, int8_t *quant_values, float scale, int32_t zp, int size,
                         int32_t min_value, int32_t max_value);
int DoQuantizeFp32ToInt8FromUint8Source(const float *real_values, int8_t *quant_values, float scale, int32_t zp,
                                        int size, int32_t min_value, int32_t max_value);
#ifdef ENABLE_ARM64
void Fp32ToInt8_arm64(const float *real_values, int8_t *quant_values, float scale, int32_t zp, int size,
                      int32_t min_value, int32_t max_value);
void Int8ToFp32_arm64(const int8_t *quant_values, float *dst, float scale, int32_t zp, int size);
#endif
int DoDequantizeUInt8ToFp32(const uint8_t *quant_values, float *real_values, float scale, int32_t zp, int size);
int DoQuantizeFp32ToUInt8(const float *real_values, uint8_t *quant_values, float scale, int32_t zp, int size);
int Int8ToUInt8(const int8_t *quant_values, uint8_t *real_values, int size);
int UInt8ToInt8(const uint8_t *real_values, int8_t *quant_values, int size);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_INT8_QUANTDTYPECAST_H_
