/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <stdio.h>
#include "nnacl/base/gather_base.h"

int Gather(const void *input, int64_t outer_size, int64_t byte_inner_size, int64_t limit, const int *indices,
           int64_t index_num, void *output, int64_t byte_out_stride, int *error_index) {
  if (input == NULL || output == NULL || indices == NULL || error_index == NULL) {
    return NNACL_NULL_PTR;
  }
  const int8_t *int8_in = (int8_t *)input;
  int8_t *int8_out = (int8_t *)output;
  int64_t in_stride = byte_inner_size * limit;
  for (int64_t m = 0; m < outer_size; ++m) {
    int8_t *int8_out_m = int8_out;
    for (int64_t i = 0; i < index_num; ++i) {
      int index = indices[i];
      index = index < 0 ? index + limit : index;
      if (index < 0 || index >= limit) {
        *error_index = index;
        return NNACL_GATHER_INDICES_VALUE_INVALID;
      } else {
        memcpy(int8_out_m, int8_in + index * byte_inner_size, byte_inner_size);
      }
      int8_out_m += byte_inner_size;
    }
    int8_in += in_stride;
    int8_out += byte_out_stride;
  }
  return NNACL_OK;
}
