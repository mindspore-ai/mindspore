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
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementAbsCoreCalc(block_size, block_num, input, output, element_size, i)            \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {  \
    MS_ST_F32(block_size, output + i, MS_ABS_F32(block_size, MS_LD_F32(block_size, input + i))); \
  }

int ElementAbs(const float *input, float *output, const int element_size) {
  int i = 0;

  // only avx512 support abs fp32 instruction
  MS_SIMD_RUN_AVX512(SimdElementAbsCoreCalc, input, output, element_size, i);
  for (; i < element_size; i++) {
    output[i] = fabsf(input[i]);
  }
  return NNACL_OK;
}

// cos
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
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementSquareCoreCalc(block_size, block_num, input, output, element_size, i)        \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) { \
    MS_FLOAT_32xN(block_num) vin = MS_LD_F32(block_size, input + i);                            \
    MS_ST_F32(block_size, output + i, MS_MUL_F32(block_size, vin, vin));                        \
  }

int ElementSquare(const float *input, float *output, const int element_size) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementSquareCoreCalc, input, output, element_size, i);
  for (; i < element_size; i++) {
    output[i] = input[i] * input[i];
  }
  return NNACL_OK;
}

// Sqrt
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementSqrtCoreCalc(block_size, block_num, input, output, element_size, i)            \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {   \
    MS_ST_F32(block_size, output + i, MS_SQRT_F32(block_size, MS_LD_F32(block_size, input + i))); \
  }

int ElementSqrt(const float *input, float *output, const int element_size) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementSqrtCoreCalc, input, output, element_size, i);
  for (; i < element_size; i++) {
    if (input[i] < 0) {
      return NNACL_ERRCODE_SQRT_NEGATIVE;
    }
    output[i] = sqrtf(input[i]);
  }
  return NNACL_OK;
}

// rsqrt
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementRsqrtCoreCalc(block_size, block_num, input, output, element_size, i)            \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {    \
    MS_ST_F32(block_size, output + i, MS_RSQRT_F32(block_size, MS_LD_F32(block_size, input + i))); \
  }

int ElementRsqrt(const float *input, float *output, const int element_size) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementRsqrtCoreCalc, input, output, element_size, i);
  for (; i < element_size; i++) {
    if (input[i] < 0) {
      return NNACL_ERRCODE_RSQRT_NEGATIVE;
    }
    output[i] = 1.f / sqrtf(input[i]);
  }
  return NNACL_OK;
}

// sin:
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementSinCoreCalc(block_size, block_num, input, output, element_size, i)            \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {  \
    MS_ST_F32(block_size, output + i, MS_SIN_F32(block_size, MS_LD_F32(block_size, input + i))); \
  }

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
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementRoundCoreCalc(block_size, block_num, input, output, element_size, i)            \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {    \
    MS_ST_F32(block_size, output + i, MS_ROUND_F32(block_size, MS_LD_F32(block_size, input + i))); \
  }

int ElementRound(const float *input, float *output, const int element_size) {
  int i = 0;

  // avx512 do not support round instruction
  MS_SIMD_RUN_AVX(SimdElementRoundCoreCalc, input, output, element_size, i);
  MS_SIMD_RUN_SSE(SimdElementRoundCoreCalc, input, output, element_size, i);
  for (; i < element_size; i++) {
    output[i] = roundf(input[i]);
  }
  return NNACL_OK;
}

// floor:
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementFloorCoreCalc(block_size, block_num, input, output, element_size, i)            \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {    \
    MS_ST_F32(block_size, output + i, MS_FLOOR_F32(block_size, MS_LD_F32(block_size, input + i))); \
  }

int ElementFloor(const float *input, float *output, const int element_size) {
  int i = 0;

  MS_SIMD_RUN_X86_NO_SCALAR(SimdElementFloorCoreCalc, input, output, element_size, i);
  for (; i < element_size; i++) {
    output[i] = floorf(input[i]);
  }
  return NNACL_OK;
}

// ceil
// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementCeilCoreCalc(block_size, block_num, input, output, element_size, i)            \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {   \
    MS_ST_F32(block_size, output + i, MS_CEIL_F32(block_size, MS_LD_F32(block_size, input + i))); \
  }

int ElementCeil(const float *input, float *output, const int element_size) {
  int i = 0;

  MS_SIMD_RUN_X86_NO_SCALAR(SimdElementCeilCoreCalc, input, output, element_size, i);
  for (; i < element_size; ++i) {
    output[i] = ceilf(input[i]);
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementNegativeCoreCalc(block_size, block_num, input, output, element_size, i)                \
  for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {           \
    MS_ST_F32(block_size, output + i, MS_MUL_N_F32(block_size, MS_LD_F32(block_size, input + i), -1.0f)); \
  }

int ElementNegative(const float *input, float *output, const int element_size) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementNegativeCoreCalc, input, output, element_size, i);
  for (; i < element_size; ++i) {
    output[i] = -input[i];
  }
  return NNACL_OK;
}

#define SimdElementReciprocalCoreCalc(block_size, block_num, input, output, element_size, i)                         \
  do {                                                                                                               \
    MS_FLOAT_32xN(block_num) num1_##block_num = MS_MOVN_F32(block_size, 1.0f);                                       \
    for (int block_max_size = element_size - block_num + 1; i < block_max_size; i += block_num) {                    \
      MS_ST_F32(block_size, output + i, MS_DIV_F32(block_size, num1_##block_num, MS_LD_F32(block_size, input + i))); \
    }                                                                                                                \
  } while (0)

int ElementReciprocal(const float *input, float *output, const int element_size) {
  int i = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementReciprocalCoreCalc, input, output, element_size, i);
  for (; i < element_size; ++i) {
    if (input[i] == 0.0f) {
      return NNACL_ERR;
    }
    output[i] = 1.f / input[i];
  }
  return NNACL_OK;
}

// Erf
int ElementErf(const float *input, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = erff(input[i]);
  }
  return NNACL_OK;
}
