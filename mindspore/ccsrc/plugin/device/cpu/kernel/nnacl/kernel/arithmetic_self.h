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

#ifndef NNACL_KERNEL_ARITHMETIC_SELF_H_
#define NNACL_KERNEL_ARITHMETIC_SELF_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"

typedef struct ArithmeticSelfFunction {
  int primitive_type_;
  int (*func_)(const float *input, float *output, const int element_size);
  int (*func_bool_)(const bool *input, bool *output, const int element_size);
  int (*func_int_)(const int *input, int *output, const int element_size);
  int (*func_float_bool_)(const float *input, bool *output, const int element_size);
} ArithmeticSelfFunction;

typedef struct ArithmeticSelfF16Function {
  int primitive_type_;
#ifdef ENABLE_FP16
  int (*func_)(const float16_t *input, float16_t *output, int element_size);
#endif
} ArithmeticSelfF16Function;

typedef struct ArithmeticSelfStruct {
  KernelBase base_;
  int op_type_;
  ArithmeticSelfFunction function_;
  ArithmeticSelfF16Function f16_function_;
} ArithmeticSelfStruct;

KernelBase *CreateArithmeticSelf(OpParameter *param, int data_type);

#endif  // NNACL_KERNEL_ARITHMETIC_SELF_H_
