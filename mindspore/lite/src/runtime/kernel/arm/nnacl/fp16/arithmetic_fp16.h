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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_ARITHMETIC_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_ARITHMETIC_FP16_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/arithmetic_common.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int ElementOptAddFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param);
int ElementOptSubFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param);
int ElementOptMulFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size,
                      ArithmeticParameter *param);
int ElementMulFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
int ElementMulReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
int ElementMulRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);

int ElementAddFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
int ElementAddReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
int ElementAddRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);

int ElementSubFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
int ElementSubReluFp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);
int ElementSubRelu6Fp16(float16_t *input0, float16_t *input1, float16_t *output, int element_size);

void TileDimensionsFp16(float16_t *data0, float16_t *data1, float16_t *tile_data0, float16_t *tile_data1,
                        ArithmeticParameter *param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP16_ARITHMETIC_FP16_H_
