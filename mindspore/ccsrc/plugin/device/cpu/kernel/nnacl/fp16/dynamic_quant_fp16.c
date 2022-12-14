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
  CalculateMinMaxCount8Fp16(data, count_8, real_min, real_max);
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
