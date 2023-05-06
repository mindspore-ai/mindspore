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
  if (count == 0) {
    return;
  }
#ifndef ENABLE_ARM64
  for (int i = 0; i < count; ++i) {
    *real_min = data[i] < *real_min ? data[i] : *real_min;
    *real_max = data[i] > *real_max ? data[i] : *real_max;
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
    *real_min = data[i] < *real_min ? data[i] : *real_min;
    *real_max = data[i] > *real_max ? data[i] : *real_max;
  }
#endif
}

void CalculateChannelRowMinMax(const float *data, int count, float *real_min, float *real_max, int row_length) {
  if (row_length == 0) {
    return;
  }
  int channel_total = count / row_length;
  for (int i = 0; i < channel_total; i++) {
    CalculateMinMaxFp32(data + i * row_length, row_length, real_min + i, real_max + i);
  }
}

void CalculateChannelColMinMax(const float *data, int count, float *real_min, float *real_max, int row_length) {
  if (row_length == 0) {
    return;
  }
  int row_total = count / row_length;
  for (int r = 0; r < row_total; r++) {
    const float *data_current = data + r * row_length;
    for (int c = 0; c < row_length; c++) {
      float *real_min_channel = real_min + c;
      float *real_max_channel = real_max + c;
      if (data_current[c] < *real_min_channel) {
        *real_min_channel = data_current[c];
      }
      if (data_current[c] > *real_max_channel) {
        *real_max_channel = data_current[c];
      }
    }
  }
}
