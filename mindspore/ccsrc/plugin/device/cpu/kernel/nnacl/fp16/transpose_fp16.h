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

#ifndef MINDSPORE_NNACL_FP16_TRANSPOSE_FP16_H_
#define MINDSPORE_NNACL_FP16_TRANSPOSE_FP16_H_

#include "nnacl/op_base.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/transpose.h"

#ifdef __cplusplus
extern "C" {
#endif
void TransposeDimsFp16(const float16_t *in_data, float16_t *out_data, const int *output_shape,
                       const TransposeParameter *param, int task_id, int thread_num);
int DoTransposeFp16(const float16_t *in_data, float16_t *out_data, const int *output_shape,
                    const TransposeParameter *param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_TRANSPOSE_FP16_H_
