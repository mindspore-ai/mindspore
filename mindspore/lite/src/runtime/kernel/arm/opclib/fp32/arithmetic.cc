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

#include "src/runtime/kernel/arm/opclib/fp32/arithmetic.h"

int ElementMul(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vmulq_f32(vin0, vin1);
    vst1q_f32(output, vout);
#else
    output[0] = input0[0] * input1[0];
    output[1] = input0[1] * input1[1];
    output[2] = input0[2] * input1[2];
    output[3] = input0[3] * input1[3];
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] * input1[index];
  }

  return OPCLIB_OK;
}

int BroadcastMul(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementMul(tile_input0, tile_input1, output, element_size);
}

int ElementAdd(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vaddq_f32(vin0, vin1);
    vst1q_f32(output, vout);
#else
    output[0] = input0[0] + input1[0];
    output[1] = input0[1] + input1[1];
    output[2] = input0[2] + input1[2];
    output[3] = input0[3] + input1[3];
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] + input1[index];
  }
  return OPCLIB_OK;
}

int ElementAddInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] + input1[i];
  }
  return OPCLIB_OK;
}

int BroadcastAdd(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementAdd(tile_input0, tile_input1, output, element_size);
}

int BroadcastAddInt8(int8_t *input0, int8_t *input1, int8_t *tile_input0, int8_t *tile_input1, int8_t *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensionsInt8(input0, input1, tile_input0, tile_input1, param);
  return ElementAddInt8(tile_input0, tile_input1, output, element_size);
}

int ElementSub(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vsubq_f32(vin0, vin1);
    vst1q_f32(output, vout);
#else
    output[0] = input0[0] - input1[0];
    output[1] = input0[1] - input1[1];
    output[2] = input0[2] - input1[2];
    output[3] = input0[3] - input1[3];
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] - input1[index];
  }
  return OPCLIB_OK;
}

int BroadcastSub(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementSub(tile_input0, tile_input1, output, element_size);
}

// todo c=a/b,if(b==0)
int ElementDiv(float *input0, float *input1, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input1[i] == 0) {
      return OPCLIB_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = input0[i] / input1[i];
  }
  return OPCLIB_OK;
}

int BroadcastDiv(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementDiv(tile_input0, tile_input1, output, element_size);
}

int ElementFloorMod(float *input0, float *input1, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input1[i] == 0) {
      return OPCLIB_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = input0[i] - floorf(input0[i] / input1[i]) * input1[i];
  }
  return OPCLIB_OK;
}

int BroadcastFloorMod(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementFloorMod(tile_input0, tile_input1, output, element_size);
}

int ElementFloorDiv(float *input0, float *input1, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input1[i] == 0) {
      return OPCLIB_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = floorf(input0[i] / input1[i]);
  }
  return OPCLIB_OK;
}

int BroadcastFloorDiv(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementFloorDiv(tile_input0, tile_input1, output, element_size);
}

int ElementLogicalAnd(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vandq_f32(vin0, vin1);
    vst1q_f32(output, vout);
#else
    output[0] = (float)((bool)(input0[0]) & (bool)(input1[0]));
    output[1] = (float)((bool)(input0[1]) & (bool)(input1[1]));
    output[2] = (float)((bool)(input0[2]) & (bool)(input1[2]));
    output[3] = (float)((bool)(input0[3]) & (bool)(input1[3]));
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)((bool)(input0[index]) & (bool)(input1[index]));
  }
  return OPCLIB_OK;
}

int ElementSquaredDifference(float *input0, float *input1, float *output, int element_size) {
  ElementSub(input0, input1, output, element_size);
  return ElementMul(output, output, output, element_size);
}

int BroadcastSquaredDifference(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                               int element_size, ArithmeticParameter *param) {
  BroadcastSub(input0, input1, tile_input0, tile_input1, output, element_size, param);
  return ElementMul(output, output, output, element_size);
}

int BroadcastLogicalAnd(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                        int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLogicalAnd(tile_input0, tile_input1, output, element_size);
}

int ElementLogicalOr(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vorrq_f32(vin0, vin1);
    vst1q_f32(output, vout);
#else
    output[0] = (float)((bool)(input0[0]) | (bool)(input1[0]));
    output[1] = (float)((bool)(input0[1]) | (bool)(input1[1]));
    output[2] = (float)((bool)(input0[2]) | (bool)(input1[2]));
    output[3] = (float)((bool)(input0[3]) | (bool)(input1[3]));
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)((bool)(input0[index]) | (bool)(input1[index]));
  }
  return OPCLIB_OK;
}

int BroadcastLogicalOr(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                       int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLogicalOr(tile_input0, tile_input1, output, element_size);
}

int ElementMaximum(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vmaxq_f32(vin0, vin1);
    vst1q_f32(output, vout);
#else
    output[0] = input0[0] > input1[0] ? input0[0] : input1[0];
    output[1] = input0[1] > input1[1] ? input0[1] : input1[1];
    output[2] = input0[2] > input1[2] ? input0[2] : input1[2];
    output[3] = input0[3] > input1[3] ? input0[3] : input1[3];
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] > input1[index] ? input0[index] : input1[index];
  }
  return OPCLIB_OK;
}

int BroadcastMaximum(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementMaximum(tile_input0, tile_input1, output, element_size);
}

int ElementMinimum(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vminq_f32(vin0, vin1);
    vst1q_f32(output, vout);
#else
    output[0] = input0[0] > input1[0] ? input1[0] : input0[0];
    output[1] = input0[1] > input1[1] ? input1[1] : input0[1];
    output[2] = input0[2] > input1[2] ? input1[2] : input0[2];
    output[3] = input0[3] > input1[3] ? input1[3] : input0[3];
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = input0[index] > input1[index] ? input1[index] : input0[index];
  }
  return OPCLIB_OK;
}

int BroadcastMinimum(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementMinimum(tile_input0, tile_input1, output, element_size);
}

int ElementNotEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef USE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vceqq_fp32(vin0, vin1), vfalse, vtrue);
    vst1q_f32(output, vout);
#else
    output[0] = (float)(input0[0] != input1[0]);
    output[1] = (float)(input0[1] != input1[1]);
    output[2] = (float)(input0[2] != input1[2]);
    output[3] = (float)(input0[3] != input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] != input1[index]);
  }
  return OPCLIB_OK;
}

int BroadcastNotEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementNotEqual(tile_input0, tile_input1, output, element_size);
}

int ElementEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef USE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vceqq_fp32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output, vout);
#else
    output[0] = (float)(input0[0] == input1[0]);
    output[1] = (float)(input0[1] == input1[1]);
    output[2] = (float)(input0[2] == input1[2]);
    output[3] = (float)(input0[3] == input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] == input1[index]);
  }
  return OPCLIB_OK;
}

int BroadcastEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                   int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementEqual(tile_input0, tile_input1, output, element_size);
}

int ElementLess(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef USE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcltq_fp32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output, vout);
#else
    output[0] = (float)(input0[0] < input1[0]);
    output[1] = (float)(input0[1] < input1[1]);
    output[2] = (float)(input0[2] < input1[2]);
    output[3] = (float)(input0[3] < input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] < input1[index]);
  }
  return OPCLIB_OK;
}

int BroadcastLess(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                  ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLess(tile_input0, tile_input1, output, element_size);
}

int ElementLessEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef USE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcleq_fp32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output, vout);
#else
    output[0] = (float)(input0[0] <= input1[0]);
    output[1] = (float)(input0[1] <= input1[1]);
    output[2] = (float)(input0[2] <= input1[2]);
    output[3] = (float)(input0[3] <= input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] <= input1[index]);
  }
  return OPCLIB_OK;
}

int BroadcastLessEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                       int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLessEqual(tile_input0, tile_input1, output, element_size);
}

int ElementGreater(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef USE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcgtq_fp32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output, vout);
#else
    output[0] = (float)(input0[0] > input1[0]);
    output[1] = (float)(input0[1] > input1[1]);
    output[2] = (float)(input0[2] > input1[2]);
    output[3] = (float)(input0[3] > input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] > input1[index]);
  }
  return OPCLIB_OK;
}

int BroadcastGreater(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementGreater(tile_input0, tile_input1, output, element_size);
}

int ElementGreaterEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef USE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef USE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcgeq_fp32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output, vout);
#else
    output[0] = (float)(input0[0] >= input1[0]);
    output[1] = (float)(input0[1] >= input1[1]);
    output[2] = (float)(input0[2] >= input1[2]);
    output[3] = (float)(input0[3] >= input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] >= input1[index]);
  }
  return OPCLIB_OK;
}

int BroadcastGreaterEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                          int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementGreaterEqual(tile_input0, tile_input1, output, element_size);
}

