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

void Stack(char **inputs, char *output, size_t input_num, size_t copy_size, size_t outter_size) {
  size_t in_offset = 0;
  size_t out_offset = 0;
  for (size_t i = 0; i < outter_size; ++i) {
    for (size_t j = 0; j < input_num; ++j) {
      memcpy(output + out_offset, inputs[j] + in_offset, copy_size);
      out_offset += copy_size;
    }
    in_offset += copy_size;
  }
}
