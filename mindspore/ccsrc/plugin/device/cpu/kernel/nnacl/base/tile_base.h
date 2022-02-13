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

#ifndef MINDSPORE_NNACL_BASE_TILE_H_
#define MINDSPORE_NNACL_BASE_TILE_H_

#include "nnacl/op_base.h"
#define MAX_TILE_DIM_SIZE 8
typedef struct TileParameter {
  // primitive parameter
  OpParameter op_parameter_;
  int multiples_[MAX_TILE_DIM_SIZE];
  int dims_[MAX_TILE_DIM_SIZE];
  size_t dims_size_;
  size_t multiples_size_;

  // shape correlative
  int in_shape_[MAX_TILE_DIM_SIZE];
  int out_shape_[MAX_TILE_DIM_SIZE];
  int in_strides_[MAX_TILE_DIM_SIZE];
  int out_strides_[MAX_TILE_DIM_SIZE];

  // other parameter
  int in_dim_;
  size_t data_size_;
  size_t fast_outer_size_;
  size_t fast_stride_;
  size_t fast_multiple_;
} TileParameter;

#ifdef __cplusplus
extern "C" {
#endif
void Tile(void *input_data, void *output_data, const TileParameter *parameter);
void TileSimple(void *input_data, void *output_data, size_t begin, size_t end, const TileParameter *parameter);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_BASE_TILE_H_
