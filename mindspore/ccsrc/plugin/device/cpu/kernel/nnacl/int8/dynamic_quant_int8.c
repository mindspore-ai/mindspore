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

#include "nnacl/int8/dynamic_quant_int8.h"
void CalculateMinMaxFp32(const float *data, int count, float *real_min, float *real_max) {
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
  volatile int count_4 = DOWN_ROUND(count, C4NUM);
  asm volatile(
    "mov x4, %[data]\n"          // reload data
    "mov w5, %w[count_4]\n"      // reload count
    "ld1 {v31.4s}, [x4]\n"       // min
    "ld1 {v30.4s}, [x4], #16\n"  // max
    "subs w5, w5, #4\n"
    "ble 1f\n"

    "0:\n"
    "ld1 {v0.4s}, [x4], #16\n"
    "fmin v31.4s, v31.4s, v0.4s\n"
    "fmax v30.4s, v30.4s, v0.4s\n"
    "subs w5, w5, #4\n"
    "bgt 0b\n"

    "1:\n"
    "fminv s6, v31.4s\n"
    "fmaxv s7, v30.4s\n"

    "str s6, [%[real_min]]\n"
    "str s7, [%[real_max]]\n"

    :
    : [ data ] "r"(data), [ count_4 ] "r"(count_4), [ real_min ] "r"(real_min), [ real_max ] "r"(real_max)
    : "x4", "w5", "s6", "s7", "v0", "v30", "v31");
  for (int i = count_4; i < count; ++i) {
    if (data[i] < *real_min) {
      *real_min = data[i];
    }
    if (data[i] > *real_max) {
      *real_max = data[i];
    }
  }
#endif
}
