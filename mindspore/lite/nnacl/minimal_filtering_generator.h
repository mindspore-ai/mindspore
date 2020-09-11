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

#ifndef MINDSPORE_LITE_NNACL_MINIMAL_FILTERING_GENERATOR_H_
#define MINDSPORE_LITE_NNACL_MINIMAL_FILTERING_GENERATOR_H_

#ifdef ENABLE_ARM
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
void Polynomial(float *interval, float *m, int degree);

void DiagonalPlusMatrix(float *matrix, float *diagonal_matrix, int degree);

void ResidueMatrix(float *interval, float *b, int row, int col);

void LT(float *poly_array, float *matrix_lt, int n);

void T(float *poly_array, float *matrix_t, int n);

void B(float *poly_array, float *matrix_b, int in_unit);

void GenerateIntervalArray(float *array, float interval, int degree);

void MatrixTranspose(float *matrix, float *trans_matrix, int row, int col);

void MatrixMultiply(const float *matrix_a, const float *matrix_b, float *matrix_c, int m, int k, int n);

void CookToomFilter(float *matrix_a, float *matrix_at, float *matrix_b, float *matrix_bt, float *matrix_g,
                    float *matrix_gt, float coefficient, int out_unit, int filter_size);

#ifdef ENABLE_ARM
void MatrixMultiplyVec(const float32x4_t *matrix_a, const float32x4_t *matrix_b, float32x4_t *matrix_c,
                       const float *bias, int m, int k, int n);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_MINIMAL_FILTERING_GENERATOR_H_
