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

#ifdef ENABLE_FP16
void InitExpMMFp16TileCount(int *row_tile, int *deep_tile, int *col_tile) {
  *row_tile = C16NUM;
  *col_tile = C12NUM;
  *deep_tile = 1;
}

void PackExpMatmulInFp16(void *dst_ptr, void *src_ptr, size_t row, size_t deep, size_t src_stride) {
  float16_t *dst = (float16_t *)dst_ptr;
  float16_t *src = (float16_t *)src_ptr;

  for (int d = 0; d < deep; d++) {
    int deep_mod8 = d % C8NUM;
    int deep_div8 = d / C8NUM;
    for (int r = 0; r < row; r++) {
      dst[d * row + r] = src[deep_div8 * src_stride * C8NUM + r * C8NUM + deep_mod8];
    }
  }
}

void ExpMatmulFp16(void *c_ptr, const void *a_ptr, const void *b_ptr, const void *bias_ptr, size_t row, size_t deep,
                   size_t col, size_t dst_stride, float min, float max) {
  float16_t *fp16_c = (float16_t *)c_ptr;
  float16_t *fp16_a = (float16_t *)a_ptr;
  float16_t *fp16_b = (float16_t *)b_ptr;
  float16_t *fp16_bias = (float16_t *)bias_ptr;

  /* dst_stride : total_row * pack */
  for (size_t r = 0; r < row; r++) {
    for (size_t c = 0; c < col; c++) {
      float dst = 0;
      size_t c_div8 = c / C8NUM;
      size_t c_mod8 = c % C8NUM;
      for (size_t d = 0; d < deep; d++) {
        size_t a_index = d * row + r;
        size_t b_index = c_div8 * deep * C8NUM + d * C8NUM + c_mod8;
        dst += fp16_a[a_index] * fp16_b[b_index];
      }

      if (fp16_bias != NULL) {
        dst += fp16_bias[c];
      }
      dst = MSMIN(dst, max);
      dst = MSMAX(dst, min);
      size_t dst_index = c_div8 * dst_stride + r * C8NUM + c_mod8;
      fp16_c[dst_index] = dst;
    }
  }
}

void ExpMatmulRemainFp16(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                         size_t dst_stride, float min, float max) {
  return ExpMatmulFp16(c_ptr, a_ptr, b_ptr, bias_ptr, row, deep, col, dst_stride, min, max);
}
void ExpMatMulBlockFp16(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                        size_t dst_stride, float min, float max) {
  return ExpMatmulFp16(c_ptr, a_ptr, b_ptr, bias_ptr, row, deep, col, dst_stride, min, max);
}
#endif

void InitExpMMFp32TileCount(int *row_tile, int *deep_tile, int *col_tile) {
  *row_tile = C16NUM;
  *col_tile = C4NUM;
  *deep_tile = 1;
}

void PackExpMatmulIn(void *dst_ptr, void *src_ptr, size_t row, size_t deep, size_t src_stride) {
  float *dst = (float *)dst_ptr;
  float *src = (float *)src_ptr;

  for (int d = 0; d < deep; d++) {
    int deep_mod4 = d % C4NUM;
    int deep_div4 = d / C4NUM;
    for (int r = 0; r < row; r++) {
      dst[d * row + r] = src[deep_div4 * src_stride * C4NUM + r * C4NUM + deep_mod4];
    }
  }
}

void ExpMatmul(float *c_ptr, const float *a_ptr, const float *b_ptr, const float *bias, size_t row, size_t deep,
               size_t col, size_t dst_stride, float min, float max) {
  /* dst_stride : total_row * pack */
  for (size_t r = 0; r < row; r++) {
    for (size_t c = 0; c < col; c++) {
      float dst = 0;
      size_t c_div4 = c / C4NUM;
      size_t c_mod4 = c % C4NUM;
      for (size_t d = 0; d < deep; d++) {
        size_t a_index = d * row + r;
        size_t b_index = c_div4 * deep * C4NUM + d * C4NUM + c_mod4;
        dst += a_ptr[a_index] * b_ptr[b_index];
      }

      if (bias != NULL) {
        dst += bias[c];
      }
      dst = MSMIN(dst, max);
      dst = MSMAX(dst, min);
      size_t dst_index = c_div4 * dst_stride + r * C4NUM + c_mod4;
      c_ptr[dst_index] = dst;
    }
  }
}

void ExpMatMulBlock(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                    size_t dst_stride, float min, float max) {
  float *c = (float *)c_ptr;
  float *a = (float *)a_ptr;
  float *b = (float *)b_ptr;
  float *bias = (float *)bias_ptr;
  return ExpMatmul(c, a, b, bias, row, deep, col, dst_stride, min, max);
}

void ExpMatmulRemain(void *c_ptr, void *a_ptr, void *b_ptr, void *bias_ptr, size_t row, size_t deep, size_t col,
                     size_t dst_stride, float min, float max) {
  float *c = (float *)c_ptr;
  float *a = (float *)a_ptr;
  float *b = (float *)b_ptr;
  float *bias = (float *)bias_ptr;
  return ExpMatmul(c, a, b, bias, row, deep, col, dst_stride, min, max);
}

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
