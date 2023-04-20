/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef NNACL_KERNEL_TILE_H_
#define NNACL_KERNEL_TILE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"

typedef struct TileStruct {
  KernelBase base_;
  bool one_dim_tile_;
  bool resize_done_;
  int dims_[MAX_SHAPE_SIZE];
  size_t dims_size_;
  uint8_t *input_addr_;
  uint8_t *output_addr_;

  int multiples_[MAX_SHAPE_SIZE];
  int in_shape_[MAX_SHAPE_SIZE];
  int out_shape_[MAX_SHAPE_SIZE];
  int in_strides_[MAX_SHAPE_SIZE];
  int out_strides_[MAX_SHAPE_SIZE];

  int in_dim_;
  size_t data_size_;
  size_t fast_outer_size_;
  size_t fast_stride_;
  size_t fast_multiple_;
} TileStruct;

KernelBase *CreateTile(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_TILE_H_
