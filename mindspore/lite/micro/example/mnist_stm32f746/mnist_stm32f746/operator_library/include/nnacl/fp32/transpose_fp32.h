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

#ifndef MINDSPORE_LITE_NNACL_FP32_TRANSPOSE_H_
#define MINDSPORE_LITE_NNACL_FP32_TRANSPOSE_H_

#include <string.h>
#include "nnacl/transpose.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif

int DoTransposeFp32(const float *in_data, float *out_data, const int *output_shape, TransposeParameter *param);
void TransposeDimsFp32(const float *in_data, float *out_data, const int *output_shape, int *size, int *position,
                       TransposeParameter *transpose_param, int task_id, int thread_num);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_TRANSPOSE_H_
