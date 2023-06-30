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

#ifndef NNACL_KERNEL_CONCAT_H_
#define NNACL_KERNEL_CONCAT_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"

typedef struct ConcatBlockBoundaryInfo {
  int begin_input_;      // input-index of upper boundary
  int end_input_;        // input-index of lower boundary.
  int64_t begin_point_;  // offset of begin-input.
  int64_t end_point_;    // required size of end-input.
} ConcatBlockBoundaryInfo;

typedef struct ConcatStruct {
  KernelBase base_;
  int64_t outer_size_;
  uint8_t *output_;
  TypeIdC data_type_;

  bool *is_with_data_;    /* size = in_tensor_size */
  uint8_t **inputs_ptr_;  /* size = in_tensor_size */
  int64_t *inner_sizes_;  // byte-inner-size (including axis) of each input and the last one is output's.

  ConcatBlockBoundaryInfo block_boundary_infos_[MAX_THREAD_NUM]; /* dynamic block size */
  int64_t block_splits_[MAX_THREAD_NUM];                         /* dynamic block size */
  size_t block_size_;                                            /* dynamic block size = actual thread number */
} ConcatStruct;

KernelBase *CreateConcat(OpParameter *param, int data_type);
int DoConcat(ConcatStruct *concat, int task_id);
int ConcatPepare(KernelBase *self);
int ConcatRelease(KernelBase *self);
int ConcatResize(KernelBase *self);

#endif  // NNACL_KERNEL_CONCAT_H_
