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
#ifndef NNACL_FP32_PAD_FP32_H_
#define NNACL_FP32_PAD_FP32_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <memory.h>
#include <float.h>
#include "nnacl/op_base.h"
#include "nnacl/pad_parameter.h"

int GetInputFlattenIndex(int out_flatten_index, const int32_t *input_shape, const int *in_strides,
                         const int *out_strides, const int *paddings, int mirror_offset);
#ifdef __cplusplus
extern "C" {
#endif
void Pad(const float *input_data, float *output_data, const int32_t *input_shape, const int32_t *output_shape,
         const int32_t *paddings, int tid, int thread_num);
void MirrorPad(const float *input_data, float *output_data, const int32_t *input_shape, const int *in_strides,
               const int *out_strides, const int *paddings, int mirror_offset, int begin, int end);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_PAD_H_
