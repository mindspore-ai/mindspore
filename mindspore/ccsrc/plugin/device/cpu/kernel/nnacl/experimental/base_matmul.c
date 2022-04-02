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

#include "nnacl/experimental/base_matmul.h"
#include "nnacl/experimental/fp32_funcs.h"

typedef struct BaseMatmulStru {
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
} BaseMatmulStru;

int BaseMatmulRun(void *param, int task_id, float lhs_scale, float rhs_scale) {
  BaseMatmulStru *mm = (BaseMatmulStru *)param;
  if (mm == NULL) {
    return -1;
  }

  size_t pack_uint = mm->base->funcs->pack * mm->base->funcs->byte;

  for (size_t i = task_id; i < mm->row_unit; i += mm->thread_num) {
    int xStart = i * mm->row_tile;
    uint8_t *a = mm->a_ptr + xStart * pack_uint;
    uint8_t *tmp = mm->tmp_ptr + mm->row_tile * mm->deep * task_id * mm->base->funcs->byte;
    mm->base->funcs->PackLeft(tmp, a, mm->row_tile, mm->deep, mm->row);
    mm->base->funcs->Matmul(mm->c_ptr + xStart * pack_uint, tmp, mm->b_ptr, mm->bias, mm->row_tile, mm->deep, mm->col,
                            mm->row * mm->base->funcs->pack, mm->min, mm->max);
  }
  return 0;
}

void BaseMatmul(uint8_t *a_ptr, uint8_t *b_ptr, uint8_t *bias, uint8_t *c_ptr, int row, int deep, int col,
                ActType act_type, int thread_num, KernelBase *base) {
  BaseMatmulStru basemm;
  if (a_ptr == NULL || b_ptr == NULL || c_ptr == NULL) {
    return;
  }
  basemm.base = base;
  basemm.deep = deep;
  basemm.col = col;
  basemm.row = row;
  basemm.a_ptr = a_ptr;
  basemm.b_ptr = b_ptr;
  basemm.c_ptr = c_ptr;
  basemm.bias = bias;
  basemm.thread_num = thread_num;

  int byte = basemm.base->funcs->byte;
  int pack = basemm.base->funcs->pack;
  int row_tile, deep_tile, col_tile;
  basemm.base->funcs->InitMatmulTileCount(&row_tile, &deep_tile, &col_tile);

  basemm.row_tile = row_tile;
  if (row_tile == 0) {
    return;
  }
  basemm.row_unit = row / row_tile;

  if (bias != NULL || act_type != ActType_No) {
    GetPostParameters(act_type, &basemm.min, &basemm.max);
  }

  basemm.tmp_ptr = (uint8_t *)basemm.base->env->alloc(basemm.base->env->allocator,
                                                      thread_num * UP_ROUND(deep, deep_tile) * row_tile * byte);
  basemm.base->env->parallelLaunch(basemm.base->env->threadPool, BaseMatmulRun, &basemm, thread_num);

  size_t row_remain = row - basemm.row_unit * row_tile;
  if (row_remain != 0) {
    int32_t start_row = basemm.row_unit * row_tile;
    uint8_t *a_remain_ptr = a_ptr + start_row * pack * byte;
    basemm.base->funcs->PackLeft(basemm.tmp_ptr, a_remain_ptr, row_remain, deep, row);
    basemm.base->funcs->MatMulRes(c_ptr + start_row * pack * byte, basemm.tmp_ptr, b_ptr, bias, row_remain, basemm.deep,
                                  basemm.col, basemm.row * basemm.base->funcs->pack, basemm.min, basemm.max);
  }

  basemm.base->env->free(basemm.base->env->allocator, basemm.tmp_ptr);
  return;
}
