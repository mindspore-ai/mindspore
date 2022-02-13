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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_ARITHMETIC_GRAD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_ARITHMETIC_GRAD_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void ElementDivNegSquareFp16(const float16_t *nom, const float16_t *denom, float16_t *output, int element_size);
void ElementMulAndDivNegSquareFp16(const float16_t *a, const float16_t *b, const float16_t *denom, float16_t *output,
                                   int element_size);
int ElementAbsGradFp16(const float16_t *in1, const float16_t *in2, float16_t *out, int element_size);
void MaximumByAxesFp16(const float16_t *input0, const float16_t *input1, const float16_t *dy, const int *input0_dims,
                       const int *input1_dims, const int *dy_dims, float16_t *output0, float16_t *output1,
                       int num_dims);
void MinimumByAxesFp16(const float16_t *input0, const float16_t *input1, const float16_t *dy, const int *input0_dims,
                       const int *input1_dims, const int *dy_dims, float16_t *output0, float16_t *output1,
                       int num_dims);
int ElementSqrtGradFp16(const float16_t *in1, const float16_t *in2, float16_t *out, const int element_size);
int ElementRsqrtGradFp16(const float16_t *in1, const float16_t *in2, float16_t *out, const int element_size);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_ARITHMETIC_GRAD_H_
