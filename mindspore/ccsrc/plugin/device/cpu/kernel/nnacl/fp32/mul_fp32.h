/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_FP32_MUL_H_
#define MINDSPORE_NNACL_FP32_MUL_H_

#include "nnacl/op_base.h"
#include "nnacl/base/arithmetic_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int ElementMul(const float *in0, const float *in1, float *out, int size);
int ElementMulRelu(const float *in0, const float *in1, float *out, int size);
int ElementMulRelu6(const float *in0, const float *in1, float *out, int size);
int ElementMulInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size);
int ElementMulReluInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size);
int ElementMulRelu6Int(const int32_t *in0, const int32_t *in1, int32_t *out, int size);
int ElementOptMul(const float *in0, const float *in1, float *out, int size, bool first_scalar);
int ElementOptMulRelu(const float *in0, const float *in1, float *out, int size, bool first_scalar);
int ElementOptMulRelu6(const float *in0, const float *in1, float *out, int size, bool first_scalar);
int ElementOptMulInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar);
int ElementOptMulReluInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar);
int ElementOptMulRelu6Int(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar);
int BroadcastMul(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_MUL_H_
