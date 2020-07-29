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
#include "src/runtime/kernel/arm/opclib/fp32/arithmetic_self.h"

// abs:
int ElementAbs(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = fabsf(input[i]);
  }
  return OPCLIB_OK;
}

// cos:
int ElementCos(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = cosf(input[i]);
  }
  return OPCLIB_OK;
}

// exp:
int ElementExp(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = expf(input[i]);
  }
  return OPCLIB_OK;
}

// log:
int ElementLog(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] <= 0) {
      return OPCLIB_ERRCODE_LOG_NEGATIVE_OR_ZERO;
    }
    output[i] = logf(input[i]);
  }
  return OPCLIB_OK;
}

// Square
int ElementSquare(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input[i] * input[i];
  }
  return OPCLIB_OK;
}

// Sqrt
int ElementSqrt(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] < 0) {
      return OPCLIB_ERRCODE_SQRT_NEGATIVE;
    }
    output[i] = sqrtf(input[i]);
  }
  return OPCLIB_OK;
}

// rsqrt
int ElementRsqrt(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] <= 0) {
      return OPCLIB_ERRCODE_RSQRT_NEGATIVE_OR_ZERO;
    }
    output[i] = 1.f / sqrtf(input[i]);
  }
  return OPCLIB_OK;
}

// sin:
int ElementSin(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = sinf(input[i]);
  }
  return OPCLIB_OK;
}

// logical_not:
int ElementLogicalNot(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = (float)(!((bool)(input[i])));
  }
  return OPCLIB_OK;
}

// round:
int ElementRound(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = round(input[i]);
  }
  return OPCLIB_OK;
}

// floor:
int ElementFloor(float *input, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = floorf(input[i]);
  }
  return OPCLIB_OK;
}

int ElementCeil(float *input, float *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = ceil(input[i]);
  }
  return OPCLIB_OK;
}
