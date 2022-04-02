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
#ifndef MINDSPORE_NNACL_EXPERIMENT_CORE_FUNCS_H_
#define MINDSPORE_NNACL_EXPERIMENT_CORE_FUNCS_H_

typedef struct CoreFuncs {
  int pack;
  int byte;
  void (*InitMatmulTileCount)(int *row_tile, int *deep_tile, int *col_tile);
  void (*PackNcX)(const void *src, void *dst, int batch, int plane, int channel);
  void (*UnPackNcX)(const void *src, void *dst, int batch, int plane, int channel);
  void (*PackLeft)(void *dst, void *src, size_t row, size_t deep, size_t src_stride);
  void (*PackRight)(const void *src, void *dst, int batch, int plane, int channel);
  void (*Matmul)(void *c_ptr, void *a_ptr, void *b_ptr, void *bias, size_t row, size_t deep, size_t col,
                 size_t dst_stride, float min, float max);
  void (*MatMulRes)(void *c_ptr, void *a_ptr, void *b_ptr, void *bias, size_t row, size_t deep, size_t col,
                    size_t dst_stride, float min, float max);
} CoreFuncs;

#endif  // MINDSPORE_NNACL_EXPERIMENT_CORE_FUNCS_H_
