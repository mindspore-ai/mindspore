/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_INT8_DYNAMIC_MATMUL_H_
#define MINDSPORE_NNACL_INT8_DYNAMIC_MATMUL_H_

#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/matmul_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void PackInput2Col4x4(const int8_t *src_input, int8_t *packed_input, int row, int col, int row_stride);
void PackInput4x4(const int8_t *src_input, int8_t *packed_input, size_t input_channel, size_t plane_size);
void DynamicMatmul4x16x4AIWI(const int8_t *a, const int8_t *b, const float *bias, float *dst, int row, int col,
                             int deep, int deep16, size_t stride, int input_zp, float input_scale,
                             const float *filter_scale, const int filter_zp, bool filter_per_channel);
#ifdef ENABLE_ARM64
void DynamicMatmulSdot4x4x16AIWI(const int8_t *a, const int8_t *b, float *out, size_t deep4, float *multi_scales,
                                 float *bias, size_t row, size_t col, size_t stride);
#else
void DynamicMatmul4x4x16AIWI(const int8_t *a, const int8_t *b, const float *bias, float *dst, int row, int col,
                             int deep4, size_t stride, float input_scale, const float *filter_scale,
                             bool filter_per_channel);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_INT8_DYNAMIC_MATMUL_H_
