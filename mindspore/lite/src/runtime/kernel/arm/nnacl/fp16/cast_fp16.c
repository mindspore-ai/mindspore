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
#include "nnacl/fp16/cast_fp16.h"

void Float32ToFloat16(const float *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

void Float16ToFloat32(const float16_t *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float)input[i];
  }
}
