
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

#ifndef NNACL_KERNEL_PAD_H_
#define NNACL_KERNEL_PAD_H_

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/pad_parameter.h"

typedef struct MirrorPadBlock {
  int out_offset_;
  int out_stride_[DEFAULT_PAD_NDIMS];
  int size_[DEFAULT_PAD_NDIMS];
} MirrorPadBlock;

typedef struct PadStruct {
  KernelBase base_;
  int data_type_;
  int mirror_offset_;
  float constant_value_;
  int pad_mode_;
  int paddings_[MAX_PAD_SIZE];
  int paddings_size_;
  int in_[DEFAULT_PAD_NDIMS];
  int out_[DEFAULT_PAD_NDIMS];
  int in_strides_[DEFAULT_PAD_NDIMS];
  int out_strides_[DEFAULT_PAD_NDIMS];
  MirrorPadBlock *mirror_pad_block_;
  int mirror_pad_block_size_;
} PadStruct;

KernelBase *CreatePad(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_PAD_H_
