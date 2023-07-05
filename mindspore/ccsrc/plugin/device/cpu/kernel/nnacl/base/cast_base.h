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
#ifndef NNACL_BASE_CAST_BASE_H_
#define NNACL_BASE_CAST_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/nnacl_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void BoolToFloat32(const bool *input, float *output, int number);

void Uint8ToFloat32(const uint8_t *input, float *output, int number);

void Int32ToFloat32(const int32_t *input, float *output, int number);

void Int64ToFloat32(const int64_t *input, float *output, int number);

#ifdef ENABLE_FP16
void Int64ToFp16(const int64_t *input, float16_t *output, int number);

void Int32ToFp16(const int32_t *input, float16_t *output, int number);

void BoolToFp16(const bool *input, float16_t *output, int number);

void Uint8ToFp16(const uint8_t *input, float16_t *output, int number);

void Float32ToFp16(const float *input, float16_t *output, int number);

void Fp16ToFloat32(const float16_t *input, float *output, int number);
#else
void Fp16ToFloat32(const uint16_t *input, float *output, int number);

void Float32ToFp16(const float *input, uint16_t *output, int number);
#endif

void Float32ToInt32(const float *input, int32_t *output, int number);

void Float32ToInt64(const float *input, int64_t *output, int number);

void Int32ToInt64(const int32_t *input, int64_t *output, int number);

void Int64ToInt32(const int64_t *input, int32_t *output, int number);

void Float32ToInt16(const float *input, int16_t *output, int number);

void BoolToInt32(const bool *input, int32_t *output, int number);

void Float32ToBool(const float *input, bool *output, int number);

void Float32ToUint8(const float *input, uint8_t *output, int number);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_BASE_CAST_BASE_H_
