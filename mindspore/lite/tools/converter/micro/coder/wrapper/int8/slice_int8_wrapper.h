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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_SLICE_INT8_WRAPPER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_SLICE_INT8_WRAPPER_H_

#include <stdint.h>
#include "nnacl/slice_parameter.h"
#include "nnacl/kernel/slice.h"

typedef struct SliceArgs {
  int8_t *input_data_;
  int8_t *output_data_;
  SliceStruct *slice_struct_;
  SliceQuantArg *quant_args_;
  int thread_num_;
} SliceArgs;

int SliceInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale);
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_WRAPPER_INT8_SLICE_INT8_WRAPPER_H_
