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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_ACTIVATION_GRAD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_ACTIVATION_GRAD_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/int8/fixed_point.h"

#ifdef __cplusplus
extern "C" {
#endif

int ReluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst);
int Relu6Fp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst);
int LReluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst, float16_t alpha);
int SigmoidFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst);
int TanhFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst);
int HSwishFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst);
int HSigmoidFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst);
int EluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst, float16_t alpha);
int GeluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_ACTIVATION_GRAD_H_
