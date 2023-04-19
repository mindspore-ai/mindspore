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

#ifndef NNACL_KERNEL_TRANSPOSE_H_
#define NNACL_KERNEL_TRANSPOSE_H_

#include "nnacl/op_base.h"
#include "nnacl/kernel.h"
#include "nnacl/transpose_parameter.h"

typedef struct TransposeStruct {
  KernelBase base_;
  bool is_valid_;
  int num_axes_;
  int data_num_;
  int perm_[MAX_TRANSPOSE_DIM_SIZE];
  int perm_size_;
  int in_shape_[MAX_TRANSPOSE_DIM_SIZE]; /* after shape optimize */
  int in_shape_size_;
  int out_shape_[MAX_TRANSPOSE_DIM_SIZE];
  int strides_[MAX_TRANSPOSE_DIM_SIZE];
  int out_strides_[MAX_TRANSPOSE_DIM_SIZE];

  int opt_perm_[PERM_NUM_THREE];  // only valid when opt_run_ is true
  bool opt_run_;                  // only true when perm is [1, 0] or [0, 2, 1]

  int (*compute_)(const void *src, void *dst, const int *out_shape, int *perm, int *strides, int *out_strides,
                  int data_size, int num_axes);
  void (*nhwc2nchw_)(const void *src, void *dst, int b, int hw, int c, int task_id, int thread);
  void (*optimize_)(const void *src, void *dst, const int *out_shape, int *perm, int *strides, int *out_strides,
                    int num_axes, int task_id, int thread);
} TransposeStruct;

KernelBase *CreateTranspose(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_TRANSPOSE_H_
