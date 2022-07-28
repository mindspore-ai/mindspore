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

#include "nnacl/experimental/ms_core.h"
#include "nnacl/op_base.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/exp_fp32.h"

void GetPostParameters(ActType act, float *min, float *max) {
#define RELU6_VALUE 6.0f
#define RELU_VALUE 0.0f
  *min = -FLT_MAX;
  *max = FLT_MAX;

  if (act == ActType_Relu) {
    *min = RELU_VALUE;
  }
  if (act == ActType_Relu6) {
    *min = RELU_VALUE;
    *max = RELU6_VALUE;
  }
  return;
}

void InitExpMMFp32TileCount(int *row_tile, int *deep_tile, int *col_tile) {
  *row_tile = C16NUM;
  *col_tile = C4NUM;
  *deep_tile = 1;
}

void PackExpMatmulIn(void *dst_ptr, void *src_ptr, size_t row, size_t deep, size_t src_stride) {
  /* src_stride : total row */
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

static void ExpMatmul(float *c_ptr, const float *a_ptr, const float *b_ptr, const float *bias, size_t row, size_t deep,
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

void InitOptMatmulTile(int *row_tile, int *col_tile) {
  *row_tile = C12NUM;
  *col_tile = C8NUM;
}

void InitCore(CoreFuncs *funcs_) {
  funcs_->pack = C4NUM;
  funcs_->byte = sizeof(float);
  funcs_->ExpMatmulTile = InitExpMMFp32TileCount;
  funcs_->PackNcX = PackNCHWToNC4HW4Fp32;
  funcs_->UnPackNcX = PackNC4HW4ToNCHWFp32;
  funcs_->ExpMatmulPackIn = PackExpMatmulIn;
  funcs_->ExpMatmulBlock = ExpMatMulBlock;
  funcs_->ExpMatMulRemain = ExpMatmulRemain;
  funcs_->ExpFusion = ExpFusionFp32;
  funcs_->OptMatmulTile = InitOptMatmulTile;
  funcs_->PostParam = GetPostParameters;
}
