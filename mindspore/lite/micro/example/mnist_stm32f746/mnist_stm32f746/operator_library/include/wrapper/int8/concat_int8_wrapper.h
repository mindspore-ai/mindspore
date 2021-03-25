/*
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

#ifndef MINDSPORE_LITE_MICRO_INT8_CONCAT_WRAPPER_INT8_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_INT8_CONCAT_WRAPPER_INT8_WRAPPER_H_

#include "nnacl/errorcode.h"
#include "nnacl/concat_parameter.h"
#include "nnacl/int8/concat_int8.h"

typedef struct {
  int8_t **inputs_;
  int8_t *output_;
  ConcatParameter *para_;
  int axis_;
  int64_t before_axis_size_;
  int64_t count_unit_;
} ConcatInt8Args;

int ConcatInt8Run(void *cdata, int task_id);

#endif  // MINDSPORE_LITE_MICRO_INT8_CONCAT_WRAPPER_INT8_WRAPPER_H_
