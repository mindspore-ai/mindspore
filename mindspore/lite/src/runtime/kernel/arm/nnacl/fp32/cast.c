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

#include "nnacl/fp32/cast.h"

void Uint8ToFloat32(const uint8_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

void Uint8ToInt8(const uint8_t *input, int8_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int8_t)(input[i] - 128);
  }
}

void Int8ToUint8(const int8_t *input, uint8_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (uint8_t)(input[i] + 128);
  }
}

void Int32ToFloat32(const int32_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}

void Float32ToInt32(const float *input, int32_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int32_t)input[i];
  }
}
