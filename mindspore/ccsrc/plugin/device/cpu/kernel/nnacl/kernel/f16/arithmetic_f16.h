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

#ifndef NNACL_KERNEL_F16_ARITHMETIC_F16_H_
#define NNACL_KERNEL_F16_ARITHMETIC_F16_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/kernel/arithmetic.h"

typedef struct ArithmeticF16Funcions {
  int primitive_type_;
  int activation_type_;
  int (*compute_)(const float16_t *in1, const float16_t *in2, float16_t *out, int ele);
  int (*optimzie_)(const float16_t *in1, const float16_t *in2, float16_t *out, int ele, bool first_scalar);
} ArithmeticF16Funcions;

typedef struct ArithmeticF16Struct {
  ArithmeticStruct arithmetic_;
  ArithmeticF16Funcions functions_;
  void *tmp_buffer_[THREE_TENSOR]; /* in_size + out_size */
} ArithmeticF16Struct;

KernelBase *CreateArithmeticF16(OpParameter *param, int data_type);
int ArithmeticF16Resize(KernelBase *self);
int ArithmeticF16Compute(KernelBase *self);

#endif  // MINDSPORE_NNACL_KERNEL_F16_ARITHMETIC_F16_H_
