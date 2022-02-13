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

#ifndef MINDSPORE_NNACL_FP16_MATRIX_FP16_H_
#define MINDSPORE_NNACL_FP16_MATRIX_FP16_H_

#include <arm_neon.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
void MatrixMultiplyFp16(const float16_t *matrix_a, const float16_t *matrix_b, float16_t *matrix_c, int m, int k, int n);

void MatrixMultiplyVecFp16(const float16x8_t *matrix_a, const float16x8_t *matrix_b, float16x8_t *matrix_c,
                           const float16_t *bias, int m, int k, int n);
void MatrixMultiplyWinogradFp16(const float16_t *matix_a, const float16_t *matrix_b, float16_t *matrix_c, int m, int k,
                                int n, int in_channel);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_MATRIX_FP16_H_
