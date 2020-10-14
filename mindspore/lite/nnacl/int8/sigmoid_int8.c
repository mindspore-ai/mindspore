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

#include "nnacl/int8/sigmoid_int8.h"

int SigmoidInt8(const int8_t *src, int length, int8_t *dst, int8_t *table) {
  for (int i = 0; i < length; i++) {
    const int8_t input_value = src[i];
    uint8_t index = (uint8_t)input_value;
    dst[i] = table[index];
  }
  return 0;
}
