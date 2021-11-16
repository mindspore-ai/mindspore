/**
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
#include "nnacl/fp16/unique_fp16.h"

int FindFp16(const float16_t *array, int len, float16_t target) {
  for (int i = 0; i < len; ++i) {
    if (array[i] == target) {
      return i;
    }
  }
  return -1;
}

void UniqueFp16(const float16_t *input, int input_len, float16_t *output0, int *output0_len, int *output1) {
  *output0_len = 0;
  for (int i = 0; i < input_len; i++) {
    int idx = FindFp16(output0, *output0_len, input[i]);
    if (idx != -1) {
      *output1++ = idx;
    } else {
      output0[(*output0_len)++] = input[i];
      *output1++ = *output0_len - 1;
    }
  }
}
