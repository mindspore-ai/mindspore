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

#include "nnacl/kernel/matmul_experimental.h"

int ExpMatmulRun(void *param, int task_id, float lhs_scale, float rhs_scale) {
  MatmulExpStru *matmul = (MatmulExpStru *)param;
  if (matmul == NULL) {
    return -1;
  }

  size_t pack_uint = matmul->base->funcs->pack * matmul->base->funcs->byte;

  for (size_t i = task_id; i < matmul->row_unit; i += matmul->thread_num) {
    int xStart = i * matmul->row_tile;
    uint8_t *a = matmul->a_ptr + xStart * pack_uint;
    uint8_t *tmp = matmul->tmp_ptr + matmul->row_tile * matmul->deep * task_id * matmul->base->funcs->byte;
    matmul->base->funcs->ExpMatmulPackIn(tmp, a, matmul->row_tile, matmul->deep, matmul->row);
    matmul->base->funcs->ExpMatmulBlock(matmul->c_ptr + xStart * pack_uint, tmp, matmul->b_ptr, matmul->bias,
                                        matmul->row_tile, matmul->deep, matmul->col,
                                        matmul->row * matmul->base->funcs->pack, matmul->min, matmul->max);
  }
  return 0;
}

void ExperimentalMatmul(uint8_t *a_ptr, uint8_t *b_ptr, uint8_t *bias, uint8_t *c_ptr, MatmulExpStru *matmul) {
  if (a_ptr == NULL || b_ptr == NULL || c_ptr == NULL) {
    return;
  }

  matmul->a_ptr = a_ptr;
  matmul->b_ptr = b_ptr;
  matmul->c_ptr = c_ptr;
  matmul->bias = bias;

  int byte = matmul->base->funcs->byte;
  int pack = matmul->base->funcs->pack;
  int row_tile, deep_tile, col_tile;
  matmul->base->funcs->ExpMatmulTile(&row_tile, &deep_tile, &col_tile);

  matmul->row_tile = row_tile;
  if (row_tile == 0) {
    return;
  }
  matmul->row_unit = matmul->row / row_tile;

  size_t tmp_size = matmul->thread_num * UP_ROUND(matmul->deep, deep_tile) * row_tile * byte;
  matmul->tmp_ptr = (uint8_t *)matmul->base->env->alloc(matmul->base->env->allocator, tmp_size);
  matmul->base->env->parallelLaunch(matmul->base->env->threadPool, ExpMatmulRun, matmul, matmul->thread_num);

  size_t row_remain = matmul->row - matmul->row_unit * row_tile;
  if (row_remain != 0) {
    int32_t start_row = matmul->row_unit * row_tile;
    uint8_t *a_remain_ptr = a_ptr + start_row * pack * byte;
    matmul->base->funcs->ExpMatmulPackIn(matmul->tmp_ptr, a_remain_ptr, row_remain, matmul->deep, matmul->row);
    matmul->base->funcs->ExpMatMulRemain(c_ptr + start_row * pack * byte, matmul->tmp_ptr, b_ptr, bias, row_remain,
                                         matmul->deep, matmul->col, matmul->row * pack, matmul->min, matmul->max);
  }

  matmul->base->env->free(matmul->base->env->allocator, matmul->tmp_ptr);
  return;
}
