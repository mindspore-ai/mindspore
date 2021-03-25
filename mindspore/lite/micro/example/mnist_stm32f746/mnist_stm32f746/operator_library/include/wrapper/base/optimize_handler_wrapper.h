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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_OPTIMIZE_HANDLER_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_OPTIMIZE_HANDLER_WRAPPER_H_

#include "nnacl/op_base.h"

#ifdef ENABLE_ARM64
void IndirectGemmInt8_optimize_handler(int8_t *dst, const int8_t *src, const int8_t *weight, const int32_t *bias,
                                       size_t ksize, size_t ic4, size_t output_channel, size_t offset,
                                       const int32_t *input_sum, size_t act_min, size_t act_max, size_t out_zp,
                                       int32_t *out_multiplier, int32_t *shift_before, int32_t *shift_after,
                                       size_t asymmetric, size_t per_channel, size_t per_channel_offset);
void MatMulR4Int8_optimize_handler(const int8_t *a, const int8_t *b, int *dst, int row4, int col4, int deep16,
                                   const int *input_sum, const int *bias);

void MatMulRInt8_optimize_handler(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                  size_t stride, const int32_t *input_sum, const int32_t *bias, int32_t *left_shift,
                                  int32_t *right_shift, int32_t *multiplier, int32_t output_zp, int32_t mini,
                                  int32_t maxi, size_t per_channel);
void MatMulDpInt8_optimize_handler(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                   size_t stride, const int32_t *input_sum, const int32_t *bias, int32_t *left_shift,
                                   int32_t *right_shift, int32_t *multiplier, int32_t output_zp, int32_t mini,
                                   int32_t maxi, size_t per_channel, int32_t *filter_zp);
#endif

#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_OPTIMIZE_HANDLER_WRAPPER_H_
