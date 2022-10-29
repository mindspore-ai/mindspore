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

#include "nnacl/fp16/dynamic_quant_fp16.h"

void CalculateMinMaxFp16(const float16_t *data, int count, float16_t *real_min, float16_t *real_max) {
#ifndef ENABLE_ARM64
  for (int i = 0; i < count; ++i) {
    if (data[i] < *real_min) {
      *real_min = data[i];
    }
    if (data[i] > *real_max) {
      *real_max = data[i];
    }
  }
#else
  // avoid to compile optimize.
  volatile int count_8 = DOWN_ROUND(count, C8NUM);
  asm volatile(
    "mov x4, %[data]\n"          // reload data
    "mov w5, %w[count_8]\n"      // reload count
    "ld1 {v31.8h}, [x4]\n"       // min
    "ld1 {v30.8h}, [x4], #16\n"  // max
    "subs w5, w5, #8\n"
    "ble 1f\n"

    "0:\n"
    "ld1 {v0.8h}, [x4], #16\n"
    "smin v31.8h, v31.8h, v0.8h\n"
    "smax v30.8h, v30.8h, v0.8h\n"
    "subs w5, w5, #8\n"
    "bgt 0b\n"

    "1:\n"
    "sminv h6, v31.8h\n"
    "smaxv h7, v30.8h\n"

    "str h6, [%[real_min]]\n"
    "str h7, [%[real_max]]\n"

    :
    : [ data ] "r"(data), [ count_8 ] "r"(count_8), [ real_min ] "r"(real_min), [ real_max ] "r"(real_max)
    : "x4", "w5", "s6", "s7", "v0", "v30", "v31");
  for (int i = count_8; i < count; ++i) {
    if (data[i] < *real_min) {
      *real_min = data[i];
    }
    if (data[i] > *real_max) {
      *real_max = data[i];
    }
  }
#endif
}
