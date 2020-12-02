/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_NNACL_ADD_INT8_H_
#define MINDSPORE_LITE_NNACL_ADD_INT8_H_

#include "nnacl/op_base.h"

typedef struct AddQuantQrgs {
  int32_t zp_;
  int32_t left_shift_;
  int32_t right_shift_;
  int32_t multiplier_;
} AddQuantQrgs;

typedef struct AddQuantParameter {
  int left_shift_;
  int32_t min_;
  int32_t max_;

  AddQuantQrgs in0_args_;
  AddQuantQrgs in1_args_;

  int32_t out_zp_;
  int32_t out_left_shift_;
  int32_t out_right_shift_;
  int32_t out_multiplier_;
} AddQuantParameter;

#ifdef __cplusplus
extern "C" {
#endif

void AddInt8(const int8_t *input0, const int8_t *input1, int8_t *output, int size, AddQuantParameter *params);
void AddOptInt8(const int8_t *ptr_in, const int8_t element_in, int8_t *output, int size, AddQuantParameter *params,
                AddQuantQrgs *ptr_args, AddQuantQrgs *ele_args);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_ADD_INT8_H_
