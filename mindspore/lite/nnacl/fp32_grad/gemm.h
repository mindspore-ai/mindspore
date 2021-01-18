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

#ifndef MINDSPORE_LITE_NNACL_FP32_GRAD_GEMM_H_
#define MINDSPORE_LITE_NNACL_FP32_GRAD_GEMM_H_

#include <stdlib.h>
#include "nnacl/op_base.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  int ca;
  int cb;
  ActType atype;
  float *bias;
  float *mat_a;
  float *mat_b;
} GemmCb;

void GemmMatmulPlus(int ta, int tb, int M, int N, int K, float alpha, const float *mat_a, int lda, const float *mat_b,
                    int ldb, float beta, float *mat_c, int ldc, float *workspace, GemmCb *cb);
void GemmMatmul(int ta, int tb, int M, int N, int K, float alpha, const float *mat_a, int lda, const float *mat_b,
                int ldb, float beta, float *mat_c, int ldc, float *workspace);
int MatSize(int row, int col, int round);
int MatSizeTotal(int row, int col, int deep, int inc);
void AddMatrix(const float *v1, float *v2, float beta, int row, int col, int stride);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_GRAD_GEMM_H_
