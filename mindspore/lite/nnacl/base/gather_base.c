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

#include "nnacl/base/gather_base.h"

int Gather(const void *input, int outer_size, int inner_size, int limit, const int *indices, int indices_element_size,
           void *output, int data_size) {
  const int8_t *int8_in = (int8_t *)input;
  int8_t *int8_out = (int8_t *)output;

  for (int m = 0; m < outer_size; ++m) {
    const int8_t *int8_in_m = int8_in + inner_size * m * limit * data_size;
    int8_t *int8_out_m = int8_out + inner_size * m * indices_element_size * data_size;

    for (int i = 0; i < indices_element_size; ++i) {
      if (indices[i] < 0 || indices[i] > limit) {
        return NNACL_ERR;
      }
      memcpy(int8_out_m + i * inner_size * data_size, int8_in_m + indices[i] * inner_size * data_size,
             data_size * inner_size);
    }
  }
  return NNACL_OK;
}
