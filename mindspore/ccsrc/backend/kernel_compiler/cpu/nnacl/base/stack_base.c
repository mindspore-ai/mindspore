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
#include "nnacl/base/stack_base.h"

void Stack(void **inputs, void *output, size_t input_num, size_t copy_size, int outer_start, int outer_end) {
  size_t out_offset = 0;
  for (size_t i = outer_start; i < outer_end; ++i) {
    for (size_t j = 0; j < input_num; ++j) {
      memcpy((char *)output + out_offset, (char *)inputs[j] + i * copy_size, copy_size);
      out_offset += copy_size;
    }
  }
}
