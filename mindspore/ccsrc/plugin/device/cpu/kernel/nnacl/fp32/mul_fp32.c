/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/mul_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"

int BroadcastMul(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param) {
  TileDimensionsFp32(in0, in1, tile_in0, tile_in1, param);
  return ElementMul(tile_in0, tile_in1, out, size);
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementMulCoreCalc(block_size, block_num, in0, in1, out, size, index)               \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                         \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
    MS_FLOAT_32xN(block_num) vout = MS_MUL_F32(block_size, vin0, vin1);                         \
    MS_ST_F32(block_size, out + index, vout);                                                   \
  }

int ElementMul(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementMulCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = in0[index] * in1[index];
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementMulReluCoreCalc(block_size, block_num, in0, in1, out, size, index)                   \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {         \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                 \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                 \
    MS_FLOAT_32xN(block_num) vout = MS_MAX_N_F32(block_size, MS_MUL_F32(block_size, vin0, vin1), 0.0f); \
    MS_ST_F32(block_size, out + index, vout);                                                           \
  }

int ElementMulRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementMulReluCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    float res = in0[index] * in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementMulRelu6CoreCalc(block_size, block_num, in0, in1, out, size, index)                    \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {           \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                   \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                   \
    MS_FLOAT_32xN(block_num) vout =                                                                       \
      MS_MIN_N_F32(block_size, MS_MAX_N_F32(block_size, MS_MUL_F32(block_size, vin0, vin1), 0.0f), 6.0f); \
    MS_ST_F32(block_size, out + index, vout);                                                             \
  }

int ElementMulRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementMulRelu6CoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] * in1[index], 0), 6);
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementMulIntCoreCalc(block_size, block_num, in0, in1, out, size, index)            \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                         \
    MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                         \
    MS_INT_32xN(block_num) vout = MS_MUL_EPI32(block_size, vin0, vin1);                         \
    MS_ST_EPI32(block_size, out + index, vout);                                                 \
  }

int ElementMulInt(const int *in0, const int *in1, int *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementMulIntCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = in0[index] * in1[index];
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementMulReluIntCoreCalc(block_size, block_num, in0, in1, out, size, index)                  \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {           \
    MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                                   \
    MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                                   \
    MS_INT_32xN(block_num) vout = MS_MAX_N_EPI32(block_size, MS_MUL_EPI32(block_size, vin0, vin1), 0.0f); \
    MS_ST_EPI32(block_size, out + index, vout);                                                           \
  }

int ElementMulReluInt(const int *in0, const int *in1, int *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementMulReluIntCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    int res = in0[index] * in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementMulRelu6IntCoreCalc(block_size, block_num, in0, in1, out, size, index)                       \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {                 \
    MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                                         \
    MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                                         \
    MS_INT_32xN(block_num) vout =                                                                               \
      MS_MIN_N_EPI32(block_size, MS_MAX_N_EPI32(block_size, MS_MUL_EPI32(block_size, vin0, vin1), 0.0f), 6.0f); \
    MS_ST_EPI32(block_size, out + index, vout);                                                                 \
  }

int ElementMulRelu6Int(const int *in0, const int *in1, int *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementMulRelu6IntCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] * in1[index], 0), 6);
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptMulCoreCalc1(block_size, block_num, in0, in1, out, size, index)             \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_F32(block_size, in0[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
      MS_FLOAT_32xN(block_num) vout = MS_MUL_F32(block_size, vin0_opt_##block_num, vin1);         \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

#define SimdElementOptMulCoreCalc2(block_size, block_num, in0, in1, out, size, index)             \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_F32(block_size, in1[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                         \
      MS_FLOAT_32xN(block_num) vout = MS_MUL_F32(block_size, vin0, vin1_opt_##block_num);         \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

int ElementOptMul(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulCoreCalc1, in0, in1, out, size, index);

    for (; index < size; index++) {
      out[index] = in0[0] * in1[index];
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulCoreCalc2, in0, in1, out, size, index);

    for (; index < size; index++) {
      out[index] = in0[index] * in1[0];
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptMulReluCoreCalc1(block_size, block_num, in0, in1, out, size, index)         \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_F32(block_size, in0[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
      MS_FLOAT_32xN(block_num) vout =                                                             \
        MS_MAX_N_F32(block_size, MS_MUL_F32(block_size, vin0_opt_##block_num, vin1), 0.0f);       \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

#define SimdElementOptMulReluCoreCalc2(block_size, block_num, in0, in1, out, size, index)         \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_F32(block_size, in1[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                         \
      MS_FLOAT_32xN(block_num) vout =                                                             \
        MS_MAX_N_F32(block_size, MS_MUL_F32(block_size, vin0, vin1_opt_##block_num), 0.0f);       \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

int ElementOptMulRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulReluCoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] * in1[index], 0);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulReluCoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] * in1[0], 0);
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptMulRelu6CoreCalc1(block_size, block_num, in0, in1, out, size, index)                     \
  do {                                                                                                         \
    MS_FLOAT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_F32(block_size, in0[0]);                           \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {              \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                      \
      MS_FLOAT_32xN(block_num) vout = MS_MIN_N_F32(                                                            \
        block_size, MS_MAX_N_F32(block_size, MS_MUL_F32(block_size, vin0_opt_##block_num, vin1), 0.0f), 6.0f); \
      MS_ST_F32(block_size, out + index, vout);                                                                \
    }                                                                                                          \
  } while (0)

#define SimdElementOptMulRelu6CoreCalc2(block_size, block_num, in0, in1, out, size, index)                     \
  do {                                                                                                         \
    MS_FLOAT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_F32(block_size, in1[0]);                           \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {              \
      MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                      \
      MS_FLOAT_32xN(block_num) vout = MS_MIN_N_F32(                                                            \
        block_size, MS_MAX_N_F32(block_size, MS_MUL_F32(block_size, vin0, vin1_opt_##block_num), 0.0f), 6.0f); \
      MS_ST_F32(block_size, out + index, vout);                                                                \
    }                                                                                                          \
  } while (0)

int ElementOptMulRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulRelu6CoreCalc1, in0, in1, out, size, index);

    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] * in1[index], 0), 6);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulRelu6CoreCalc2, in0, in1, out, size, index);

    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] * in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptMulIntCoreCalc1(block_size, block_num, in0, in1, out, size, index)          \
  do {                                                                                            \
    MS_INT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_EPI32(block_size, in0[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                         \
      MS_INT_32xN(block_num) vout = MS_MUL_EPI32(block_size, vin0_opt_##block_num, vin1);         \
      MS_ST_EPI32(block_size, out + index, vout);                                                 \
    }                                                                                             \
  } while (0)

#define SimdElementOptMulIntCoreCalc2(block_size, block_num, in0, in1, out, size, index)          \
  do {                                                                                            \
    MS_INT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_EPI32(block_size, in1[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                         \
      MS_INT_32xN(block_num) vout = MS_MUL_EPI32(block_size, vin0, vin1_opt_##block_num);         \
      MS_ST_EPI32(block_size, out + index, vout);                                                 \
    }                                                                                             \
  } while (0)

int ElementOptMulInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulIntCoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[0] * in1[index];
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulIntCoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[index] * in1[0];
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptMulReluIntCoreCalc1(block_size, block_num, in0, in1, out, size, index)      \
  do {                                                                                            \
    MS_INT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_EPI32(block_size, in0[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                         \
      MS_INT_32xN(block_num) vout =                                                               \
        MS_MAX_N_EPI32(block_size, MS_MUL_EPI32(block_size, vin0_opt_##block_num, vin1), 0.0f);   \
      MS_ST_EPI32(block_size, out + index, vout);                                                 \
    }                                                                                             \
  } while (0)

#define SimdElementOptMulReluIntCoreCalc2(block_size, block_num, in0, in1, out, size, index)      \
  do {                                                                                            \
    MS_INT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_EPI32(block_size, in1[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                         \
      MS_INT_32xN(block_num) vout =                                                               \
        MS_MAX_N_EPI32(block_size, MS_MUL_EPI32(block_size, vin0, vin1_opt_##block_num), 0.0f);   \
      MS_ST_EPI32(block_size, out + index, vout);                                                 \
    }                                                                                             \
  } while (0)

int ElementOptMulReluInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulReluIntCoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] * in1[index], 0);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulReluIntCoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] * in1[0], 0);
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptMulRelu6IntCoreCalc1(block_size, block_num, in0, in1, out, size, index)                      \
  do {                                                                                                             \
    MS_INT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_EPI32(block_size, in0[0]);                               \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {                  \
      MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                                          \
      MS_INT_32xN(block_num) vout = MS_MIN_N_EPI32(                                                                \
        block_size, MS_MAX_N_EPI32(block_size, MS_MUL_EPI32(block_size, vin0_opt_##block_num, vin1), 0.0f), 6.0f); \
      MS_ST_EPI32(block_size, out + index, vout);                                                                  \
    }                                                                                                              \
  } while (0)

#define SimdElementOptMulRelu6IntCoreCalc2(block_size, block_num, in0, in1, out, size, index)                      \
  do {                                                                                                             \
    MS_INT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_EPI32(block_size, in1[0]);                               \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {                  \
      MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                                          \
      MS_INT_32xN(block_num) vout = MS_MIN_N_EPI32(                                                                \
        block_size, MS_MAX_N_EPI32(block_size, MS_MUL_EPI32(block_size, vin0, vin1_opt_##block_num), 0.0f), 6.0f); \
      MS_ST_EPI32(block_size, out + index, vout);                                                                  \
    }                                                                                                              \
  } while (0)

int ElementOptMulRelu6Int(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulRelu6IntCoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] * in1[index], 0), 6);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptMulRelu6IntCoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] * in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}
