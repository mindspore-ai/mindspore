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

#ifndef MINDSPORE_NNACL_EXPERIMENT_MS_CORE_H_
#define MINDSPORE_NNACL_EXPERIMENT_MS_CORE_H_

#include <float.h>
#include "nnacl/op_base.h"
#include "nnacl/exp_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CoreFuncs {
  int pack;
  int byte;
  int (*ExpFusion)(const void *src_data, void *dst_data, const ExpParameter *param, int task_id);
  void (*PackNcX)(const void *src, void *dst, int batch, int plane, int channel);
  void (*UnPackNcX)(const void *src, void *dst, int batch, int plane, int channel);
  void (*PostParam)(ActType act, float *min, float *max);

  void (*ExpMatmulTile)(int *row_tile, int *deep_tile, int *col_tile);
  void (*ExpMatmulPackIn)(void *dst, void *src, size_t row, size_t deep, size_t src_stride);
  void (*ExpMatmulBlock)(void *c_ptr, void *a_ptr, void *b_ptr, void *bias, size_t row, size_t deep, size_t col,
                         size_t dst_stride, float min, float max);
  void (*ExpMatMulRemain)(void *c_ptr, void *a_ptr, void *b_ptr, void *bias, size_t row, size_t deep, size_t col,
                          size_t dst_stride, float min, float max);

  void (*OptMatmulTile)(int *row_tile, int *col_tile);
} CoreFuncs;

/* x86 */
void InitCore(CoreFuncs *funcs_);

/* arm64 fp32 */
void InitFp32Core(CoreFuncs *funcs_);

/* arm64 fp16 */
void InitFp16Core(CoreFuncs *funcs_);

/* arm32 */
void InitArm32Core(CoreFuncs *funcs_);

/* avx */
void InitAvxCore(CoreFuncs *funcs_);

/* avx512 */
void InitAvx512Core(CoreFuncs *funcs_);

/* sse */
void InitSseCore(CoreFuncs *funcs_);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_EXPERIMENT_MS_CORE_H_
