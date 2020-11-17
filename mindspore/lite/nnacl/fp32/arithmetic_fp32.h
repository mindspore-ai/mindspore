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
#include "nnacl/arithmetic_common.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int ElementOptAdd(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param);
int ElementOptAddInt(const int *input0, const int *input1, int *output, const int element_size,
                     const ArithmeticParameter *param);
int ElementOptAddRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param);
int ElementOptAddRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param);
int ElementOptSub(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param);
int ElementOptSubRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param);
int ElementOptSubRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param);
int ElementOptMul(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param);
int ElementOptMulRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param);
int ElementOptMulRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param);
int ElementOptMulInt(const int *input0, const int *input1, int *output, const int element_size,
                     const ArithmeticParameter *param);
int ElementOptMulReluInt(const int *input0, const int *input1, int *output, const int element_size,
                         const ArithmeticParameter *param);
int ElementOptMulRelu6Int(const int *input0, const int *input1, int *output, const int element_size,
                          const ArithmeticParameter *param);
int ElementOptDiv(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param);
int ElementOptDivRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param);
int ElementOptDivRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param);
int ElementMul(const float *input0, const float *input1, float *output, const int element_size);
int ElementMulRelu(const float *input0, const float *input1, float *output, const int element_size);
int ElementMulRelu6(const float *input0, const float *input1, float *output, const int element_size);
int ElementMulInt(const int *input0, const int *input1, int *output, const int element_size);
int ElementMulReluInt(const int *input0, const int *input1, int *output, const int element_size);
int ElementMulRelu6Int(const int *input0, const int *input1, int *output, const int element_size);
int BroadcastMul(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param);

int ElementAdd(const float *input0, const float *input1, float *output, const int element_size);
int ElementAddRelu(const float *input0, const float *input1, float *output, const int element_size);
int ElementAddRelu6(const float *input0, const float *input1, float *output, const int element_size);
int ElementAddInt(const int *input0, const int *input1, int *output, const int element_size);
int BroadcastAdd(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param);
int BroadcastAddInt8(int8_t *input0, int8_t *input1, int8_t *tile_input0, int8_t *tile_input1, int8_t *output,
                     int element_size, ArithmeticParameter *param);

int ElementSub(const float *input0, const float *input1, float *output, const int element_size);
int ElementSubRelu(const float *input0, const float *input1, float *output, const int element_size);
int ElementSubRelu6(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastSub(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param);

int ElementDiv(const float *input0, const float *input1, float *output, const int element_size);
int ElementDivRelu(const float *input0, const float *input1, float *output, const int element_size);
int ElementDivRelu6(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastDiv(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param);

int ElementLogicalAnd(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastLogicalAnd(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                        int element_size, ArithmeticParameter *param);

int ElementLogicalOr(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastLogicalOr(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                       int element_size, ArithmeticParameter *param);

int ElementMaximum(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastMaximum(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param);

int ElementMinimum(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastMinimum(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param);

int ElementFloorDiv(const float *input0, const float *input1, float *output, const int element_size);
int ElementFloorDivInt(const int *input0, const int *input1, int *output, const int element_size);
int BroadcastFloorDiv(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param);

int ElementFloorMod(const float *input0, const float *input1, float *output, const int element_size);
int ElementFloorModInt(const int *input0, const int *input1, int *output, const int element_size);
int BroadcastFloorMod(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param);

int ElementSquaredDifference(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastSquaredDifference(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                               int element_size, ArithmeticParameter *param);

int ElementNotEqual(const float *input0, const float *input1, float *output, const int element_size);

int BroadcastNotEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param);

int ElementEqual(const float *input0, const float *input1, float *output, const int element_size);

int BroadcastEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                   int element_size, ArithmeticParameter *param);

int ElementLess(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastLess(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                  ArithmeticParameter *param);

int ElementLessEqual(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastLessEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                       int element_size, ArithmeticParameter *param);

int ElementGreater(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastGreater(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param);

int ElementGreaterEqual(const float *input0, const float *input1, float *output, const int element_size);
int BroadcastGreaterEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                          int element_size, ArithmeticParameter *param);

#ifdef ENABLE_NNACL_INFER_SHAPE
int ArithmeticInferShape(int **in_shape, size_t *dim_size, int *out_shape, int *in_format, int *out_format,
                         int *in_datatype, int *out_datatype, OpParameter *param);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_ARITHMETIC_H_
