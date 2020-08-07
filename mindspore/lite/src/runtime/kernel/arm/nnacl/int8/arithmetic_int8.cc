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

#include "src/runtime/kernel/arm/nnacl/int8/arithmetic_int8.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "src/runtime/kernel/arm/nnacl/errorcode.h"

int ElementNotEqual(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int index = 0; index < element_size; ++index) {
    output[index] = (int8_t)(input0[index] != input1[index]);
  }
  return NNACL_OK;
}

int ElementEqual(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int index = 0; index < element_size; ++index) {
    output[index] = (int8_t)(input0[index] == input1[index]);
  }
  return NNACL_OK;
}

int ElementLess(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int index = 0; index < element_size; ++index) {
    output[index] = (int8_t)(input0[index] < input1[index]);
  }
  return NNACL_OK;
}

int ElementLessEqual(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int index = 0; index < element_size; ++index) {
    output[index] = (int8_t)(input0[index] <= input1[index]);
  }
  return NNACL_OK;
}

int ElementGreater(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int index = 0; index < element_size; ++index) {
    output[index] = (int8_t)(input0[index] > input1[index]);
  }
  return NNACL_OK;
}

int ElementGreaterEqual(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int index = 0; index < element_size; ++index) {
    output[index] = (int8_t)(input0[index] >= input1[index]);
  }
  return NNACL_OK;
}
