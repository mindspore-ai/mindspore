/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_NNACL_FP16_ARITHMETIC_FP16_H_
#define MINDSPORE_LITE_NNACL_FP16_ARITHMETIC_FP16_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/base/arithmetic_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif

void TileOneDimensionFp16(const float16_t *inData, float16_t *outData, int dim, size_t ndim, const int *inShape,
                          const int *inStrides, const int *outStrides, const int *multiple);
void TileDimensionsFp16(const float16_t *data0, const float16_t *data1, float16_t *tile_data0, float16_t *tile_data1,
                        ArithmeticParameter *param);

int ElementOptMulFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param);
int ElementOptMulReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param);
int ElementOptMulRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param);
int ElementOptAddFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param);
int ElementOptAddReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param);
int ElementOptAddRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param);
int ElementOptSubFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param);
int ElementOptSubReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param);
int ElementOptSubRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param);
int ElementOptDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param);
int ElementOptDivReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param);
int ElementOptDivRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param);
int ElementOptFloorModFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param);
int ElementOptFloorDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           ArithmeticParameter *param);
int ElementOptLogicalAndFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                             ArithmeticParameter *param);
int ElementOptLogicalOrFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                            ArithmeticParameter *param);
int ElementOptSquaredDifferenceFp16(const float16_t *input0, const float16_t *input1, float16_t *output,
                                    int element_size, ArithmeticParameter *param);
int ElementOptMaximumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param);
int ElementOptMinimumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          ArithmeticParameter *param);
int ElementOptNotEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                           ArithmeticParameter *param);
int ElementOptEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                        ArithmeticParameter *param);
int ElementOptLessFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                       ArithmeticParameter *param);
int ElementOptLessEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                            ArithmeticParameter *param);
int ElementOptGreaterFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                          ArithmeticParameter *param);
int ElementOptGreaterEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                               ArithmeticParameter *param);

int ElementMulFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementMulReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementMulRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementAddFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementAddReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementAddRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int BroadcastAddFp16(const float16_t *in0, const float16_t *in1, float16_t *tile_in0, float16_t *tile_in1,
                     float16_t *out, int size, ArithmeticParameter *param);

int ElementSubFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementSubReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementSubRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementDivReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementDivRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementFloorModFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementFloorDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementLogicalAndFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementLogicalOrFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementSquaredDifferenceFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementMaximumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);
int ElementMinimumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size);

int ElementNotEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size);
int ElementEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size);
int ElementLessFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size);
int ElementLessEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size);
int ElementGreaterFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size);
int ElementGreaterEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_ARITHMETIC_FP16_H_
