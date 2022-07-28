/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_EXPERIMENT_MATMUL_EXPERIMENTAL_H_
#define MINDSPORE_NNACL_EXPERIMENT_MATMUL_EXPERIMENTAL_H_

#include "nnacl/kernel.h"
#include "nnacl/experimental/ms_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MatmulExpStru {
  KernelBase *base;
  size_t deep;
  size_t row;
  size_t col;
  size_t thread_num;
  uint8_t *a_ptr;
  uint8_t *b_ptr;
  uint8_t *c_ptr;
  uint8_t *bias;
  uint8_t *tmp_ptr;
  float min;
  float max;
  size_t row_unit;
  size_t row_tile;
} MatmulExpStru;

#ifdef ENABLE_FP16
void InitExpMMFp16TileCount(int *row_tile, int *deep_tile, int *col_tile);
void PackExpMatmulInFp16(void *dst_ptr, void *src_ptr, size_t row, size_t deep, size_t src_stride);
void ExpMatmulRemainFp16(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                         size_t dst_stride, float min, float max);
void ExpMatMulBlockFp16(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                        size_t dst_stride, float min, float max);
#endif

void InitExpMMFp32TileCount(int *row_tile, int *deep_tile, int *col_tile);
void PackExpMatmulIn(void *dst_ptr, void *src_ptr, size_t row, size_t deep, size_t src_stride);
void ExpMatmulRemain(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                     size_t dst_stride, float min, float max);
void ExpMatMulBlock(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                    size_t dst_stride, float min, float max);

void ExperimentalMatmul(uint8_t *a_ptr, uint8_t *b_ptr, uint8_t *bias, uint8_t *c_ptr, MatmulExpStru *matmul);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_EXPERIMENT_MATMUL_EXPERIMENTAL_H_
