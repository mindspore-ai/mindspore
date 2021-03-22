/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_ADD_INT8_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_ADD_INT8_WRAPPER_H_
#include <string.h>
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/int8/add_int8.h"
#include "nnacl/arithmetic.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  AddQuantParameter *para_;
  ArithmeticParameter *arith_para_;
  int in_size_;
  int out_size_;
  int thread_count_;
  int elements_num_;
  bool support_opt_add_;
  int8_t *input0_data_;
  int8_t *input1_data_;
  int8_t *output_data_;
} AddInt8Args;

int AddBroadcastInt8Run(void *cdata, int task_id);

int AddInt8Run(void *cdata, int task_id);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_ADD_INT8_WRAPPER_H_
