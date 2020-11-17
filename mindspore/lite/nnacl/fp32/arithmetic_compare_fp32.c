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

#include <string.h>
#include <math.h>
#include "nnacl/fp32/arithmetic_compare_fp32.h"

// equal:
int ElementEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] == input1[i];
  }
  return NNACL_OK;
}

int ElementEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] == input1[i];
  }
  return NNACL_OK;
}

// not equal
int ElementNotEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] != input1[i];
  }
  return NNACL_OK;
}

int ElementNotEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] != input1[i];
  }
  return NNACL_OK;
}

// less
int ElementLessFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] < input1[i];
  }
  return NNACL_OK;
}

int ElementLessInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] < input1[i];
  }
  return NNACL_OK;
}

// less equal
int ElementLessEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] <= input1[i];
  }
  return NNACL_OK;
}

int ElementLessEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] <= input1[i];
  }
  return NNACL_OK;
}

// greater
int ElementGreaterFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] > input1[i];
  }
  return NNACL_OK;
}

int ElementGreaterInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] > input1[i];
  }
  return NNACL_OK;
}

// greater equal
int ElementGreaterEqualFp32(const float *input0, const float *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] >= input1[i];
  }
  return NNACL_OK;
}

int ElementGreaterEqualInt32(const int32_t *input0, const int32_t *input1, uint8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] >= input1[i];
  }
  return NNACL_OK;
}
