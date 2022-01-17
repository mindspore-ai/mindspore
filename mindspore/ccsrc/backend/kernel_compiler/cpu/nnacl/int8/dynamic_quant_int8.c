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
  for (int i = 0; i < count; ++i) {
    if (data[i] < *real_min) {
      *real_min = data[i];
    }
    if (data[i] > *real_max) {
      *real_max = data[i];
    }
  }
}
