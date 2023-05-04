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

#ifndef NNACL_KERNEL_STRIDED_SLICE_H_
#define NNACL_KERNEL_STRIDED_SLICE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"

typedef struct StridedSliceStruct {
  KernelBase base_;
  TypeIdC data_type_;
  bool fast_run_;
  bool soft_copy_mode_;
  bool parallel_on_outer_;
  bool parallel_on_split_axis_;

  int split_axis_;
  int in_shape_size_;
  int begins_[MAX_SHAPE_SIZE];
  int ends_[MAX_SHAPE_SIZE];
  int strides_[MAX_SHAPE_SIZE];
  int in_shape_[MAX_SHAPE_SIZE];

  size_t inner_;
  size_t outer_;
  size_t inner_size_;
  int cal_num_per_thread_;
} StridedSliceStruct;

KernelBase *CreateStridedSlice(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_STRIDED_SLICE_H_
