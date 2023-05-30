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
#ifndef NNACL_KERNEL_GATHER_H_
#define NNACL_KERNEL_GATHER_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"

#define GATHER_BLOCK_INFOS_SIZE 32

typedef struct GatherBlockBoundaryInfo {
  int64_t begin_batch_;
  int64_t begin_index_;
  int64_t end_batch_;
  int64_t end_index_;
} GatherBlockBoundaryInfo;

typedef struct GatherStruct {
  KernelBase base_;
  int axis_;
  int limit_;
  int outer_size_;
  int indices_size_;
  int byte_inner_size_;
  int block_infos_size_;
  int *indices_data_;
  GatherBlockBoundaryInfo block_infos_[GATHER_BLOCK_INFOS_SIZE];
} GatherStruct;

KernelBase *CreateGather(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_GATHER_H_
