/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_FP32_ADD_H_
#define MINDSPORE_NNACL_FP32_ADD_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/base/arithmetic_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif

int ElementAdd(const float *in0, const float *in1, float *out, int size);
int ElementAddExt(const float *in0, const float *in1, const float alpha, float *out, int size);
int ElementOptAddExt(const float *in0, const float *in1, const float alpha, float *out, int size, bool first_scalar);
int ElementAddRelu(const float *in0, const float *in1, float *out, int size);
int ElementAddRelu6(const float *in0, const float *in1, float *out, int size);
int ElementAddInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size);
int ElementOptAdd(const float *in0, const float *in1, float *out, int size, bool first_scalar);
int ElementOptAddRelu(const float *in0, const float *in1, float *out, int size, bool first_scalar);
int ElementOptAddRelu6(const float *in0, const float *in1, float *out, int size, bool first_scalar);
int ElementOptAddInt(const int32_t *in0, const int32_t *in1, int32_t *out, int size, bool first_scalar);
int BroadcastAdd(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_ADD_H_
