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

#include "wrapper/base/optimize_handler_wrapper.h"

extern void MatMulOptR4Int8Neon64(const int8_t *a, const int8_t *b, int *dst, int row4, int col4, int deep16,
                                  const int *input_sum, const int *bias);
extern void MatmulInt8DpNeon64(const int8_t *a, const int8_t *b, int8_t *dst, int row8, int col8, int deep4,
                               const int *a_sums, const int *bias, int act_min, int act_max, int out_zp,
                               int *multiplier, int *left_shift, int *right_shift, int row, int col, int stride,
                               size_t peroc);
extern void MatmulInt8DpOpt(const int8_t *a, const int8_t *b, int8_t *dst, size_t row8, size_t col8, size_t deep4,
                            const int *a_sums, const int *bias, int act_min, int act_max, int out_zp, int *multiplier,
                            int *left_shift, int *right_shift, size_t stride, size_t peroc, int *filter_zp);

#ifdef ENABLE_ARM64
void MatMulR4Int8_optimize_handler(const int8_t *a, const int8_t *b, int *dst, int row4, int col4, int deep16,
                                   const int *input_sum, const int *bias) {
  return MatMulOptR4Int8Neon64(a, b, dst, row4, col4, deep16, input_sum, bias);
}

void MatMulRInt8_optimize_handler(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                  size_t stride, const int32_t *input_sum, const int32_t *bias, int32_t *left_shift,
                                  int32_t *right_shift, int32_t *multiplier, int32_t output_zp, int32_t mini,
                                  int32_t maxi, size_t per_channel) {
  return MatmulInt8DpNeon64(a, b, dst, UP_ROUND(row, C8NUM), UP_ROUND(col, C8NUM), deep_4, input_sum, bias, mini, maxi,
                            output_zp, multiplier, left_shift, right_shift, row, col, stride, per_channel);
}
void MatMulDpInt8_optimize_handler(const int8_t *a, const int8_t *b, int8_t *dst, size_t row, size_t col, size_t deep_4,
                                   size_t stride, const int32_t *input_sum, const int32_t *bias, int32_t *left_shift,
                                   int32_t *right_shift, int32_t *multiplier, int32_t output_zp, int32_t mini,
                                   int32_t maxi, size_t per_channel, int32_t *filter_zp) {
  return MatmulInt8DpOpt(a, b, dst, row, col, deep_4, input_sum, bias, mini, maxi, output_zp, multiplier, left_shift,
                         right_shift, stride, per_channel, filter_zp);
}
#endif
