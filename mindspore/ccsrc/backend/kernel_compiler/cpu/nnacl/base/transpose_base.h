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

#ifndef MINDSPORE_NNACL_TRANSPOSE_BASE_H_
#define MINDSPORE_NNACL_TRANSPOSE_BASE_H_

#include "nnacl/transpose.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int DoTransposeUInt8(const uint8_t *in_data, uint8_t *out_data, const int *output_shape,
                     const TransposeParameter *transpose_param);
int DoTransposeUInt16(const uint16_t *in_data, uint16_t *out_data, const int *output_shape,
                      const TransposeParameter *transpose_param);
int DoTransposeUInt32(const uint32_t *in_data, uint32_t *out_data, const int *output_shape,
                      const TransposeParameter *transpose_param);
int DoTransposeUInt64(const uint64_t *in_data, uint64_t *out_data, const int *output_shape,
                      const TransposeParameter *transpose_param);
int DoTransposeInt16(const int16_t *in_data, int16_t *out_data, const int *output_shape,
                     const TransposeParameter *transpose_param);
int DoTransposeInt32(const int32_t *in_data, int32_t *out_data, const int *output_shape,
                     const TransposeParameter *transpose_param);
int DoTransposeInt64(const int64_t *in_data, int64_t *out_data, const int *output_shape,
                     const TransposeParameter *transpose_param);
int DoTransposeBool(const bool *in_data, bool *out_data, const int *output_shape,
                    const TransposeParameter *transpose_param);

void TransposeDimsUInt8(const uint8_t *in_data, uint8_t *out_data, const int *output_shape,
                        const TransposeParameter *transpose_param, int task_id, int thread_num);
void TransposeDimsUInt16(const uint16_t *in_data, uint16_t *out_data, const int *output_shape,
                         const TransposeParameter *transpose_param, int task_id, int thread_num);
void TransposeDimsUInt32(const uint32_t *in_data, uint32_t *out_data, const int *output_shape,
                         const TransposeParameter *transpose_param, int task_id, int thread_num);
void TransposeDimsUInt64(const uint64_t *in_data, uint64_t *out_data, const int *output_shape,
                         const TransposeParameter *transpose_param, int task_id, int thread_num);
void TransposeDimsInt16(const int16_t *in_data, int16_t *out_data, const int *output_shape,
                        const TransposeParameter *transpose_param, int task_id, int thread_num);
void TransposeDimsInt32(const int32_t *in_data, int32_t *out_data, const int *output_shape,
                        const TransposeParameter *transpose_param, int task_id, int thread_num);
void TransposeDimsInt64(const int64_t *in_data, int64_t *out_data, const int *output_shape,
                        const TransposeParameter *transpose_param, int task_id, int thread_num);
void TransposeDimsBool(const bool *in_data, bool *out_data, const int *output_shape,
                       const TransposeParameter *transpose_param, int task_id, int thread_num);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_TRANSPOSE_BASE_H_
