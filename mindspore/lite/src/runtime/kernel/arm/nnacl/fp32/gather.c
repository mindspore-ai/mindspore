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

#include "nnacl/fp32/gather.h"
#include <string.h>

inline int Stride(int *shape, int rank, int index) {
  int i, stride = 1;
  for (i = index + 1; i < rank; ++i) {
    stride *= shape[i];
  }
  return stride;
}

int Gather(float *input, int outer_size, int inner_size, int limit, int *indices, int indices_element_size,
           float *output) {
  int i, m;
  for (m = 0; m < outer_size; ++m) {
    float *inputm = input + inner_size * m * limit;
    float *outputm = output + inner_size * m * indices_element_size;
    for (i = 0; i < indices_element_size; ++i) {
      if (indices[i] < 0 || indices[i] > limit) {
        return -1;
      }
      memcpy(outputm + i * inner_size, inputm + indices[i] * inner_size, sizeof(float) * inner_size);
    }
  }
  return 0;
}
