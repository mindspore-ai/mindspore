/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_SRC_OPERATOR_CPU_INCLUDE_OP_FUNC_COMM_H_
#define PREDICT_SRC_OPERATOR_CPU_INCLUDE_OP_FUNC_COMM_H_

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "src/op_common.h"
#include "include/tensor.h"

#ifdef MS_USE_NEON
#include <arm_neon.h>
#endif  // MS_USE_NEON

namespace mindspore {
namespace predict {
#ifdef __cplusplus
extern "C" {
#endif
#define CAL_STEP 4
void MSAddBias(float *dst, const float *bias, size_t planeNumber, size_t biasNumber);
void MSAddBiasRelu(float *dst, const float *bias, size_t planeNumber, size_t biasNumber);
void MSAddBiasRelu6(float *dst, const float *bias, size_t planeNumber, size_t biasNumber);
void MSPackC4Uint8(uint8_t *dst, const uint8_t *src, size_t area, size_t depth);
void MSUnpackC4(float *dst, const float *src, size_t area, size_t depth);
void MSUnpackC4Uint8(uint8_t *dst, const uint8_t *src, size_t area, size_t depth);
void MSTensorConvertNHWCToNC4HW4(float *dst, const float *src, size_t area, size_t depth);
void MSTensorConvertNC4HW4ToNHWC(float *dst, const float *src, size_t area, size_t depth);
void MSUnpackC4(float *dst, const float *src, size_t area, size_t depth);
void MSCopyC4WithStride(const float *source, float *dest, size_t srcStride, size_t dstStride, size_t count);
void MSUInt8ToInt16WithOffsetC4Common(int16_t *dst, const uint8_t *src, size_t zeroPoint, size_t sizeQuad,
                                      size_t dstStride, size_t srcStride);
void MSUInt8ToInt16WithOffsetC4Fast(int16_t *dst, const uint8_t *src, size_t zeroPoint, size_t sizeQuad,
                                    size_t depthQuad, size_t dstZStep, size_t srcZStep);

int MSPackC4(float *dst, const float *src, size_t area, size_t depth);
int NchwToNc4hw4(const Tensor *input, Tensor *output);
int Nc4hw4ToNchw(const Tensor *input, Tensor *output);
#ifdef __cplusplus
}
#endif
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_OPERATOR_CPU_INCLUDE_OP_FUNC_COMM_H_
