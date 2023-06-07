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
#ifndef NNACL_FP16_ARITHMETIC_FP16_H_
#define NNACL_FP16_ARITHMETIC_FP16_H_

#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/base/arithmetic_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif

void TileOneDimensionFp16(const void *input, void *output, int dim, size_t ndim, const int *inShape,
                          const int *inStrides, const int *outStrides, const int *multiple);
void TileDimensionsFp16(const float16_t *data0, const float16_t *data1, float16_t *tile_data0, float16_t *tile_data1,
                        ArithmeticParameter *param);

int ElementOptMulFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      bool first_scalar);
int ElementOptMulReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          bool first_scalar);
int ElementOptMulRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           bool first_scalar);
int ElementOptAddFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      bool first_scalar);
int ElementOptAddReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          bool first_scalar);
int ElementOptAddRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           bool first_scalar);
int ElementOptSubFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      bool first_scalar);
int ElementOptSubReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          bool first_scalar);
int ElementOptSubRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           bool first_scalar);
int ElementOptDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                      bool first_scalar);
int ElementOptDivReluFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          bool first_scalar);
int ElementOptDivRelu6Fp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           bool first_scalar);
int ElementOptFloorModFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           bool first_scalar);
int ElementOptFloorDivFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                           bool first_scalar);
int ElementOptLogicalAndFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                             bool first_scalar);
int ElementOptLogicalOrFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                            bool first_scalar);
int ElementOptSquaredDifferenceFp16(const float16_t *input0, const float16_t *input1, float16_t *output,
                                    int element_size, bool first_scalar);
int ElementOptMaximumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          bool first_scalar);
int ElementOptMinimumFp16(const float16_t *input0, const float16_t *input1, float16_t *output, int element_size,
                          bool first_scalar);
int ElementOptNotEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                           bool first_scalar);
int ElementOptEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                        bool first_scalar);
int ElementOptLessFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                       bool first_scalar);
int ElementOptLessEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                            bool first_scalar);
int ElementOptGreaterFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                          bool first_scalar);
int ElementOptGreaterEqualFp16(const float16_t *input0, const float16_t *input1, uint8_t *output, int element_size,
                               bool first_scalar);

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

#endif  //  NNACL_FP16_ARITHMETIC_FP16_H_
