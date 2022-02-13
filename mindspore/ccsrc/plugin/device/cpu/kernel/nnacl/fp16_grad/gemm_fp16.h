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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_GEMM_FP16_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_GEMM_FP16_H_

#include <stdlib.h>
#include "nnacl/op_base.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  int ca;
  int cb;
  ActType atype;
  float16_t *bias;
  float16_t *mat_a;
  float16_t *mat_b;
} GemmCbFp16;

void GemmMatmulFp16(int ta, int tb, int M, int N, int K, float16_t alpha, const float16_t *mat_a, int lda,
                    const float16_t *mat_b, int ldb, float16_t beta, float16_t *mat_c, int ldc, float16_t *workspace);
void GemmMatmulPlusFp16(int ta, int tb, int M, int N, int K, float16_t alpha, const float16_t *mat_a, int lda,
                        const float16_t *mat_b, int ldb, float16_t beta, float16_t *mat_c, int ldc,
                        float16_t *workspace, GemmCbFp16 *gcb);
int MatSizeFp16(int row, int col, int round);
int MatSizeTotalFp16(int row, int col, int deep, int inc);
void AddMatrixFp16(const float16_t *v1, float16_t *v2, float16_t beta, int row, int col, int stride);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_GEMM_FP16_H_
