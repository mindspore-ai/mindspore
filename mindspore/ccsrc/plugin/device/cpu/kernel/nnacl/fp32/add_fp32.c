/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/add_fp32.h"
#include "nnacl/fp32/arithmetic_fp32.h"

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptAddCoreCalc(block_size, block_num, in0, in1, out, size, index)              \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin0_##block_num = MS_MOVN_F32(block_size, in0[0]);                  \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
      MS_FLOAT_32xN(block_num) vout = MS_ADD_F32(block_size, vin0_##block_num, vin1);             \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

int ElementOptAdd(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddCoreCalc, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[0] + in1[index];
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddCoreCalc, in1, in0, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[index] + in1[0];
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptAddIntCoreCalc(block_size, block_num, in0, in1, out, size, index)           \
  do {                                                                                            \
    MS_INT_32xN(block_num) vin0_##block_num = MS_MOVN_EPI32(block_size, in0[0]);                  \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                         \
      MS_INT_32xN(block_num) vout = MS_ADD_EPI32(block_size, vin0_##block_num, vin1);             \
      MS_ST_EPI32(block_size, out + index, vout);                                                 \
    }                                                                                             \
  } while (0)

int ElementOptAddInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddIntCoreCalc, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[0] + in1[index];
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddIntCoreCalc, in1, in0, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[index] + in1[0];
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptAddReluCoreCalc(block_size, block_num, in0, in1, out, size, index)                              \
  do {                                                                                                                \
    MS_FLOAT_32xN(block_num) vin0_##block_num = MS_MOVN_F32(block_size, in0[0]);                                      \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {                     \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                             \
      MS_FLOAT_32xN(block_num) vout = MS_MAX_N_F32(block_size, MS_ADD_F32(block_size, vin0_##block_num, vin1), 0.0f); \
      MS_ST_F32(block_size, out + index, vout);                                                                       \
    }                                                                                                                 \
  } while (0)

int ElementOptAddRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddReluCoreCalc, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] + in1[index], 0);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddReluCoreCalc, in1, in0, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] + in1[0], 0);
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptAddRelu6CoreCalc(block_size, block_num, in0, in1, out, size, index)                  \
  do {                                                                                                     \
    MS_FLOAT_32xN(block_num) vin0_##block_num = MS_MOVN_F32(block_size, in0[0]);                           \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {          \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                  \
      MS_FLOAT_32xN(block_num) vout = MS_MIN_N_F32(                                                        \
        block_size, MS_MAX_N_F32(block_size, MS_ADD_F32(block_size, vin0_##block_num, vin1), 0.0f), 6.0f); \
      MS_ST_F32(block_size, out + index, vout);                                                            \
    }                                                                                                      \
  } while (0)

int ElementOptAddRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddRelu6CoreCalc, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] + in1[index], 0), 6);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptAddRelu6CoreCalc, in1, in0, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] + in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int BroadcastAdd(const float *in0, const float *in1, float *tile_in0, float *tile_in1, float *out, int size,
                 ArithmeticParameter *param) {
  TileDimensionsFp32(in0, in1, tile_in0, tile_in1, param);
  return ElementAdd(tile_in0, tile_in1, out, size);
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementAddCoreCalc(block_size, block_num, in0, in1, out, size, index)               \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                         \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
    MS_FLOAT_32xN(block_num) vout = MS_ADD_F32(block_size, vin0, vin1);                         \
    MS_ST_F32(block_size, out + index, vout);                                                   \
  }

int ElementAdd(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementAddCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = in0[index] + in1[index];
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementAddReluCoreCalc(block_size, block_num, in0, in1, out, size, index)                   \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {         \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                 \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                 \
    MS_FLOAT_32xN(block_num) vout = MS_MAX_N_F32(block_size, MS_ADD_F32(block_size, vin0, vin1), 0.0f); \
    MS_ST_F32(block_size, out + index, vout);                                                           \
  }

int ElementAddRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementAddReluCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    float res = in0[index] + in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementAddRelu6CoreCalc(block_size, block_num, in0, in1, out, size, index)                    \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {           \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                   \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                   \
    MS_FLOAT_32xN(block_num) vout =                                                                       \
      MS_MIN_N_F32(block_size, MS_MAX_N_F32(block_size, MS_ADD_F32(block_size, vin0, vin1), 0.0f), 6.0f); \
    MS_ST_F32(block_size, out + index, vout);                                                             \
  }

int ElementAddRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementAddRelu6CoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] + in1[index], 0), 6);
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementAddIntCoreCalc(block_size, block_num, in0, in1, out, size, index)            \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                         \
    MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                         \
    MS_INT_32xN(block_num) vout = MS_ADD_EPI32(block_size, vin0, vin1);                         \
    MS_ST_EPI32(block_size, out + index, vout);                                                 \
  }

int ElementAddInt(const int *in0, const int *in1, int *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementAddIntCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = in0[index] + in1[index];
  }
  return NNACL_OK;
}
