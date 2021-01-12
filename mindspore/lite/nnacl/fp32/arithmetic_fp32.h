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
#ifndef MINDSPORE_LITE_NNACL_ARITHMETIC_H_
#define MINDSPORE_LITE_NNACL_ARITHMETIC_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/base/arithmetic_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
void TileOneDimensionFp32(const float *inData, float *outData, int dim, size_t ndim, const int *inShape,
                          const int *inStrides, const int *outStrides, const int *multiple);
void TileDimensionsFp32(const float *data0, const float *data1, float *tile_data0, float *tile_data1,
                        ArithmeticParameter *param);

/* Mul */
int ElementMul(const float *in0, const float *in1, float *out, int size);
int ElementMulRelu(const float *in0, const float *in1, float *out, int size);
int ElementMulRelu6(const float *in0, const float *in1, float *out, int size);
int ElementMulInt(const int *in0, const int *in1, int *out, int size);
int ElementMulReluInt(const int *in0, const int *in1, int *out, int size);
int ElementMulRelu6Int(const int *in0, const int *in1, int *out, int size);
int ElementOptMul(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptMulRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptMulRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptMulInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int ElementOptMulReluInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int ElementOptMulRelu6Int(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int BroadcastMul(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param);

/* Add */
int ElementAdd(const float *in0, const float *in1, float *out, int size);
int ElementAddRelu(const float *in0, const float *in1, float *out, int size);
int ElementAddRelu6(const float *in0, const float *in1, float *out, int size);
int ElementAddInt(const int *in0, const int *in1, int *out, int size);
int ElementOptAdd(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptAddInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int ElementOptAddRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptAddRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int BroadcastAdd(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param);

/* Sub */
int ElementSub(const float *in0, const float *in1, float *out, int size);
int ElementSubInt(const int *in0, const int *in1, int *out, int size);
int ElementSubRelu(const float *in0, const float *in1, float *out, int size);
int ElementSubRelu6(const float *in0, const float *in1, float *out, int size);
int ElementOptSub(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptSubRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptSubRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptSubInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);

/* Div */
int ElementDiv(const float *in0, const float *in1, float *out, int size);
int ElementDivRelu(const float *in0, const float *in1, float *out, int size);
int ElementDivRelu6(const float *in0, const float *in1, float *out, int size);
int ElementOptDiv(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptDivRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptDivRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptDivInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);
int BroadcastDiv(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param);

/* logical and */
int ElementLogicalAnd(const float *in0, const float *in1, float *out, int size);
int ElementLogicalAndInt(const int *in0, const int *in1, int *out, int size);
int ElementLogicalAndBool(const bool *in0, const bool *in1, bool *out, int size);

/* logical or */
int ElementLogicalOr(const float *in0, const float *in1, float *out, int size);

/* Element Squared Difference */
int ElementSquaredDifference(const float *in0, const float *in1, float *out, int size);

/* max min */
int ElementMaximum(const float *in0, const float *in1, float *out, int size);
int ElementMinimum(const float *in0, const float *in1, float *out, int size);
int ElementMaximumInt(const int *in0, const int *in1, int *out, int size);
int ElementMinimumInt(const int *input0, const int *input1, int *output, const int element_size);

/* floor div */
int ElementFloorDiv(const float *in0, const float *in1, float *out, int size);
int ElementFloorDivInt(const int *in0, const int *in1, int *out, int size);

/* floor mod */
int ElementFloorMod(const float *in0, const float *in1, float *out, int size);
int ElementFloorModInt(const int *in0, const int *in1, int *out, int size);

/* mod */
int ElementMod(const float *in0, const float *in1, float *out, int size);
int ElementModInt(const int *in0, const int *in1, int *out, int size);
int ElementOptMod(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param);
int ElementOptModInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_ARITHMETIC_H_
