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
#include "nnacl/fp32/arithmetic_self_fp32.h"

// abs:
int ElementAbs(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = fabsf(input[i]);
  }
  return NNACL_OK;
}

// cos:
int ElementCos(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = cosf(input[i]);
  }
  return NNACL_OK;
}

// log:
int ElementLog(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] <= 0) {
      return NNACL_ERRCODE_LOG_NEGATIVE_OR_ZERO;
    }
    output[i] = logf(input[i]);
  }
  return NNACL_OK;
}

// Square
int ElementSquare(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input[i] * input[i];
  }
  return NNACL_OK;
}

// Sqrt
int ElementSqrt(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] < 0) {
      return NNACL_ERRCODE_SQRT_NEGATIVE;
    }
    output[i] = sqrtf(input[i]);
  }
  return NNACL_OK;
}

// rsqrt
int ElementRsqrt(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] < 0) {
      return NNACL_ERRCODE_RSQRT_NEGATIVE;
    }
    output[i] = 1.f / sqrtf(input[i]);
  }
  return NNACL_OK;
}

// sin:
int ElementSin(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = sinf(input[i]);
  }
  return NNACL_OK;
}

// logical_not:
int ElementLogicalNot(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = (float)(!((bool)(input[i])));
  }
  return NNACL_OK;
}

// logical_not:
int ElementLogicalNotBool(const bool *input, bool *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = !input[i];
  }
  return NNACL_OK;
}

// round:
int ElementRound(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = round(input[i]);
  }
  return NNACL_OK;
}

// floor:
int ElementFloor(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = floorf(input[i]);
  }
  return NNACL_OK;
}

int ElementCeil(const float *input, float *output, const int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = ceil(input[i]);
  }
  return NNACL_OK;
}

int ElementNegative(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; ++i) {
    output[i] = -input[i];
  }
  return NNACL_OK;
}

int ElementReciprocal(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; ++i) {
    if (input[i] == 0.0f) {
      return NNACL_ERR;
    }
    output[i] = 1.f / input[i];
  }
  return NNACL_OK;
}

int ElementErf(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = erff(input[i]);
  }
  return NNACL_OK;
}
