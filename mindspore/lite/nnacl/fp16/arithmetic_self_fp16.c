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
#include <math.h>
#include "nnacl/fp16/arithmetic_self_fp16.h"

int ElementAbsFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = fabsf(input[i]);
  }
  return NNACL_OK;
}

int ElementCosFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = cosf(input[i]);
  }
  return NNACL_OK;
}

int ElementLogFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] <= 0) {
      return NNACL_ERRCODE_LOG_NEGATIVE_OR_ZERO;
    }
    output[i] = logf(input[i]);
  }
  return NNACL_OK;
}

int ElementSquareFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input[i] * input[i];
  }
  return NNACL_OK;
}

int ElementSqrtFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input[i] < 0) {
      return NNACL_ERRCODE_SQRT_NEGATIVE;
    }
    output[i] = sqrtf(input[i]);
  }
  return NNACL_OK;
}

int ElementRsqrtFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = 1.f / sqrtf(input[i]);
  }
  return NNACL_OK;
}

int ElementSinFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = sinf(input[i]);
  }
  return NNACL_OK;
}

int ElementLogicalNotFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = (float)(!((bool)(input[i])));
  }
  return NNACL_OK;
}

int ElementRoundFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = round(input[i]);
  }
  return NNACL_OK;
}

int ElementFloorFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = floorf(input[i]);
  }
  return NNACL_OK;
}

int ElementCeilFp16(float16_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = ceil(input[i]);
  }
  return NNACL_OK;
}

int ElementNegativeFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; ++i) {
    output[i] = -input[i];
  }
  return NNACL_OK;
}

int ElementReciprocalFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; ++i) {
    if (input[i] == 0.0f) {
      return NNACL_ERR;
    }
    output[i] = 1.f / input[i];
  }
  return NNACL_OK;
}

int ElementErfFp16(float16_t *input, float16_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = erff(input[i]);
  }
  return NNACL_OK;
}
