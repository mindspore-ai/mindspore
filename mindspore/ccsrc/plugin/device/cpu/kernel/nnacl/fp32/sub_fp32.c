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
#include "nnacl/fp32/sub_fp32.h"

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptSubCoreCalc1(block_size, block_num, in0, in1, out, size, index)             \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_F32(block_size, in0[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
      MS_FLOAT_32xN(block_num) vout = MS_SUB_F32(block_size, vin0_opt_##block_num, vin1);         \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

#define SimdElementOptSubCoreCalc2(block_size, block_num, in0, in1, out, size, index)             \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_F32(block_size, in1[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                         \
      MS_FLOAT_32xN(block_num) vout = MS_SUB_F32(block_size, vin0, vin1_opt_##block_num);         \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

int ElementOptSub(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubCoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[0] - in1[index];
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubCoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[index] - in1[0];
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptSubIntCoreCalc1(block_size, block_num, in0, in1, out, size, index)          \
  do {                                                                                            \
    MS_INT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_EPI32(block_size, in0[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                         \
      MS_INT_32xN(block_num) vout = MS_SUB_EPI32(block_size, vin0_opt_##block_num, vin1);         \
      MS_ST_EPI32(block_size, out + index, vout);                                                 \
    }                                                                                             \
  } while (0)

#define SimdElementOptSubIntCoreCalc2(block_size, block_num, in0, in1, out, size, index)          \
  do {                                                                                            \
    MS_INT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_EPI32(block_size, in1[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                         \
      MS_INT_32xN(block_num) vout = MS_SUB_EPI32(block_size, vin0, vin1_opt_##block_num);         \
      MS_ST_EPI32(block_size, out + index, vout);                                                 \
    }                                                                                             \
  } while (0)

int ElementOptSubInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubIntCoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[0] - in1[index];
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubIntCoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = in0[index] - in1[0];
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptSubReluCoreCalc1(block_size, block_num, in0, in1, out, size, index)         \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_F32(block_size, in0[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
      MS_FLOAT_32xN(block_num) vout =                                                             \
        MS_MAX_N_F32(block_size, MS_SUB_F32(block_size, vin0_opt_##block_num, vin1), 0.0f);       \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

#define SimdElementOptSubReluCoreCalc2(block_size, block_num, in0, in1, out, size, index)         \
  do {                                                                                            \
    MS_FLOAT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_F32(block_size, in1[0]);              \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
      MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                         \
      MS_FLOAT_32xN(block_num) vout =                                                             \
        MS_MAX_N_F32(block_size, MS_SUB_F32(block_size, vin0, vin1_opt_##block_num), 0.0f);       \
      MS_ST_F32(block_size, out + index, vout);                                                   \
    }                                                                                             \
  } while (0)

int ElementOptSubRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubReluCoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] - in1[index], 0);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubReluCoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] - in1[0], 0);
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementOptSubRelu6CoreCalc1(block_size, block_num, in0, in1, out, size, index)                     \
  do {                                                                                                         \
    MS_FLOAT_32xN(block_num) vin0_opt_##block_num = MS_MOVN_F32(block_size, in0[0]);                           \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {              \
      MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                      \
      MS_FLOAT_32xN(block_num) vout = MS_MIN_N_F32(                                                            \
        block_size, MS_MAX_N_F32(block_size, MS_SUB_F32(block_size, vin0_opt_##block_num, vin1), 0.0f), 6.0f); \
      MS_ST_F32(block_size, out + index, vout);                                                                \
    }                                                                                                          \
  } while (0)

#define SimdElementOptSubRelu6CoreCalc2(block_size, block_num, in0, in1, out, size, index)                     \
  do {                                                                                                         \
    MS_FLOAT_32xN(block_num) vin1_opt_##block_num = MS_MOVN_F32(block_size, in1[0]);                           \
    for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {              \
      MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                      \
      MS_FLOAT_32xN(block_num) vout = MS_MIN_N_F32(                                                            \
        block_size, MS_MAX_N_F32(block_size, MS_SUB_F32(block_size, vin0, vin1_opt_##block_num), 0.0f), 6.0f); \
      MS_ST_F32(block_size, out + index, vout);                                                                \
    }                                                                                                          \
  } while (0)

int ElementOptSubRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
  int index = 0;
  if (param->in_elements_num0_ == 1) {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubRelu6CoreCalc1, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] - in1[index], 0), 6);
    }
  } else {
    MS_SIMD_RUN_NO_SCALAR(SimdElementOptSubRelu6CoreCalc2, in0, in1, out, size, index);
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] - in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementSubCoreCalc(block_size, block_num, in0, in1, out, size, index)               \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                         \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                         \
    MS_FLOAT_32xN(block_num) vout = MS_SUB_F32(block_size, vin0, vin1);                         \
    MS_ST_F32(block_size, out + index, vout);                                                   \
  }

int ElementSub(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementSubCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = in0[index] - in1[index];
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementSubIntCoreCalc(block_size, block_num, in0, in1, out, size, index)            \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) { \
    MS_INT_32xN(block_num) vin0 = MS_LD_EPI32(block_size, in0 + index);                         \
    MS_INT_32xN(block_num) vin1 = MS_LD_EPI32(block_size, in1 + index);                         \
    MS_INT_32xN(block_num) vout = MS_SUB_EPI32(block_size, vin0, vin1);                         \
    MS_ST_EPI32(block_size, out + index, vout);                                                 \
  }

int ElementSubInt(const int *in0, const int *in1, int *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementSubIntCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = in0[index] - in1[index];
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementSubReluCoreCalc(block_size, block_num, in0, in1, out, size, index)                   \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {         \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                 \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                 \
    MS_FLOAT_32xN(block_num) vout = MS_MAX_N_F32(block_size, MS_SUB_F32(block_size, vin0, vin1), 0.0f); \
    MS_ST_F32(block_size, out + index, vout);                                                           \
  }

int ElementSubRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementSubReluCoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    float res = in0[index] - in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

// 32 bits, block_size : (512/256/128/32), block_num : (16/8/4/1)
#define SimdElementSubRelu6CoreCalc(block_size, block_num, in0, in1, out, size, index)                    \
  for (int block_max_size = size - block_num + 1; index < block_max_size; index += block_num) {           \
    MS_FLOAT_32xN(block_num) vin0 = MS_LD_F32(block_size, in0 + index);                                   \
    MS_FLOAT_32xN(block_num) vin1 = MS_LD_F32(block_size, in1 + index);                                   \
    MS_FLOAT_32xN(block_num) vout =                                                                       \
      MS_MIN_N_F32(block_size, MS_MAX_N_F32(block_size, MS_SUB_F32(block_size, vin0, vin1), 0.0f), 6.0f); \
    MS_ST_F32(block_size, out + index, vout);                                                             \
  }

int ElementSubRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;

  MS_SIMD_RUN_NO_SCALAR(SimdElementSubRelu6CoreCalc, in0, in1, out, size, index);
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] - in1[index], 0), 6);
  }

  return NNACL_OK;
}
