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

#ifndef NNACL_KERNEL_ARG_MIN_MAX_H_
#define NNACL_KERNEL_ARG_MIN_MAX_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#ifdef ENABLE_ARM64
#include <arm_neon.h>
#endif

typedef struct ArgElement {
  uint32_t index_;
  union ArgData {
    int8_t i8_data_;
    int32_t i_data_;
    float f_data_;
#ifdef ENABLE_ARM
#if (!SUPPORT_NNIE) || (defined SUPPORT_34XX)
    float16_t f16_data_;
#endif
#endif
  } data_;
} ArgElement;

typedef int (*COMPARE_FUNCTION)(const void *a, const void *b);

typedef struct ArgMinMaxComputeParam {
  int32_t axis_;
  int32_t dims_size_;
  int32_t topk_;
  bool get_max_;
  bool keep_dims_;
  bool out_value_;
  int32_t in_strides_[COMM_SHAPE_SIZE];
  int32_t out_strides_[COMM_SHAPE_SIZE];
  ArgElement *arg_elements_;
} ArgMinMaxComputeParam;

typedef struct ArgMinMaxStruct {
  KernelBase base_;
  ArgMinMaxComputeParam compute_;
  bool arg_elements_alloc_;
} ArgMinMaxStruct;

KernelBase *CreateArgMinMax(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_ARG_MIN_MAX_H_
