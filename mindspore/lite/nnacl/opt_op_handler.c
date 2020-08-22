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

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void IndirectGemmInt8_24x4_dp(int8_t *dst, const int8_t *src, const int8_t *weight, const int32_t *bias,
                                     size_t ksize, size_t ic4, size_t output_channel, size_t offset,
                                     const int32_t *input_sum, size_t act_min, size_t act_max, size_t out_zp,
                                     int32_t *out_multiplier, int32_t *shift_before, int32_t *shift_after,
                                     size_t asymmetric, size_t per_channel);

extern void MatMulOptR4Int8Neon64(const int8_t *a, const int8_t *b, int *dst, int row4, int col4, int deep16,
                                  const int *input_sum, const int *bias);
#ifdef __cplusplus
}
#endif

#ifdef ENABLE_ARM64
void IndirectGemmInt8_optimize_handler(int8_t *dst, const int8_t *src, const int8_t *weight, const int32_t *bias,
                                       size_t ksize, size_t ic4, size_t output_channel, size_t offset,
                                       const int32_t *input_sum, size_t act_min, size_t act_max, size_t out_zp,
                                       int32_t *out_multiplier, int32_t *shift_before, int32_t *shift_after,
                                     size_t asymmetric, size_t per_channel) {
  return IndirectGemmInt8_24x4_dp(dst, src, weight, bias, ksize, ic4, output_channel, offset, input_sum, act_min,
                                  act_max, out_zp, out_multiplier, shift_before, shift_after, asymmetric, per_channel);
}

void MatMulR4Int8_optimize_handler(const int8_t *a, const int8_t *b, int *dst, int row4, int col4, int deep16,
                                   const int *input_sum, const int *bias) {
  return MatMulOptR4Int8Neon64(a, b, dst, row4, col4, deep16, input_sum, bias);
}
#endif
