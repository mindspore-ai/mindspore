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

#include "nnacl/fp32/arithmetic.h"
#include <math.h>

#define ACCURACY_DATA 0.00000001

int ElementOptMul(float *input0, float *input1, float *output, int element_size, ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[0] * input1[i];
    }
  } else if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] * input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] * input1[i];
    }
  }
  return NNACL_OK;
}

int ElementOptSub(float *input0, float *input1, float *output, int element_size, ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[0] - input1[i];
    }
  } else if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] - input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] - input1[i];
    }
  }
  return NNACL_OK;
}

int ElementOptAdd(float *input0, float *input1, float *output, int element_size, ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[0] + input1[i];
    }
  } else if (param->in_elements_num1_ == 1) {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] + input1[0];
    }
  } else {
    for (int i = 0; i < element_size; ++i) {
      output[i] = input0[i] + input1[i];
    }
  }
  return NNACL_OK;
}

int ElementMul(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
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

  return NNACL_OK;
}

int ElementMulRelu(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

#ifdef ENABLE_NEON
  float32x4_t zeros = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vmulq_f32(vin0, vin1);
    vout = vbslq_f32(vcgtq_f32(vout, zeros), vout, zeros);
    vst1q_f32(output, vout);
#else
    float res = input0[0] * input1[0];
    output[0] = res > 0 ? res : 0;
    res = input0[1] * input1[1];
    output[1] = res > 0 ? res : 0;
    res = input0[2] * input1[2];
    output[2] = res > 0 ? res : 0;
    res = input0[3] * input1[3];
    output[3] = res > 0 ? res : 0;
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float res = input0[index] * input1[index];
    output[index] = res > 0 ? res : 0;
  }

  return NNACL_OK;
}

int ElementMulRelu6(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

#ifdef ENABLE_NEON
  float32x4_t zeros = {0, 0, 0, 0};
  float32x4_t bounds = {6, 6, 6, 6};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vminq_f32(vmaxq_f32(vmulq_f32(vin0, vin1), zeros), bounds);
    vst1q_f32(output, vout);
#else
    output[0] = MSMIN(MSMAX(input0[0] * input1[0], 0), 6);
    output[1] = MSMIN(MSMAX(input0[1] * input1[1], 0), 6);
    output[2] = MSMIN(MSMAX(input0[2] * input1[2], 0), 6);
    output[3] = MSMIN(MSMAX(input0[3] * input1[3], 0), 6);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] * input1[index], 0), 6);
  }

  return NNACL_OK;
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
#ifdef ENABLE_NEON
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
  return NNACL_OK;
}

int ElementAddRelu(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

#ifdef ENABLE_NEON
  float32x4_t zeros = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vaddq_f32(vin0, vin1);
    vout = vbslq_f32(vcgtq_f32(vout, zeros), vout, zeros);
    vst1q_f32(output, vout);
#else
    float res = input0[0] + input1[0];
    output[0] = res > 0 ? res : 0;
    res = input0[1] + input1[1];
    output[1] = res > 0 ? res : 0;
    res = input0[2] + input1[2];
    output[2] = res > 0 ? res : 0;
    res = input0[3] + input1[3];
    output[3] = res > 0 ? res : 0;
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float res = input0[index] + input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementAddRelu6(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

#ifdef ENABLE_NEON
  float32x4_t zeros = {0, 0, 0, 0};
  float32x4_t bounds = {6, 6, 6, 6};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vminq_f32(vmaxq_f32(vaddq_f32(vin0, vin1), zeros), bounds);
    vst1q_f32(output, vout);
#else
    output[0] = MSMIN(MSMAX(input0[0] + input1[0], 0), 6);
    output[1] = MSMIN(MSMAX(input0[1] + input1[1], 0), 6);
    output[2] = MSMIN(MSMAX(input0[2] + input1[2], 0), 6);
    output[3] = MSMIN(MSMAX(input0[3] + input1[3], 0), 6);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] + input1[index], 0), 6);
  }

  return NNACL_OK;
}

int ElementAddInt8(int8_t *input0, int8_t *input1, int8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] + input1[i];
  }
  return NNACL_OK;
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
#ifdef ENABLE_NEON
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
  return NNACL_OK;
}

int ElementSubRelu(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

#ifdef ENABLE_NEON
  float32x4_t zeros = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vsubq_f32(vin0, vin1);
    vout = vbslq_f32(vcgtq_f32(vout, zeros), vout, zeros);
    vst1q_f32(output, vout);
#else
    float res = input0[0] - input1[0];
    output[0] = res > 0 ? res : 0;
    res = input0[1] - input1[1];
    output[1] = res > 0 ? res : 0;
    res = input0[2] - input1[2];
    output[2] = res > 0 ? res : 0;
    res = input0[3] - input1[3];
    output[3] = res > 0 ? res : 0;
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    float res = input0[index] - input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementSubRelu6(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

#ifdef ENABLE_NEON
  float32x4_t zeros = {0, 0, 0, 0};
  float32x4_t bounds = {6, 6, 6, 6};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vminq_f32(vmaxq_f32(vsubq_f32(vin0, vin1), zeros), bounds);
    vst1q_f32(output, vout);
#else
    output[0] = MSMIN(MSMAX(input0[0] - input1[0], 0), 6);
    output[1] = MSMIN(MSMAX(input0[1] - input1[1], 0), 6);
    output[2] = MSMIN(MSMAX(input0[2] - input1[2], 0), 6);
    output[3] = MSMIN(MSMAX(input0[3] - input1[3], 0), 6);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = MSMIN(MSMAX(input0[index] - input1[index], 0), 6);
  }

  return NNACL_OK;
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
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = input0[i] / input1[i];
  }
  return NNACL_OK;
}

int ElementDivRelu(float *input0, float *input1, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input1[i] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    float res = input0[i] / input1[i];
    output[i] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementDivRelu6(float *input0, float *input1, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input1[i] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = MSMIN(MSMAX(input0[i] / input1[i], 0), 6);
  }
  return NNACL_OK;
}

int BroadcastDiv(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                 ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementDiv(tile_input0, tile_input1, output, element_size);
}

int ElementFloorMod(float *input0, float *input1, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input1[i] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = input0[i] - floorf(input0[i] / input1[i]) * input1[i];
  }
  return NNACL_OK;
}

int BroadcastFloorMod(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementFloorMod(tile_input0, tile_input1, output, element_size);
}

int ElementFloorDiv(float *input0, float *input1, float *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    if (input1[i] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    output[i] = floorf(input0[i] / input1[i]);
  }
  return NNACL_OK;
}

int BroadcastFloorDiv(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementFloorDiv(tile_input0, tile_input1, output, element_size);
}

int ElementLogicalAnd(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;

#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
  uint32x4_t mask = vmovq_n_u32(((uint32_t)(1u << 31) - 1));
  uint32x4_t zeros = {0, 0, 0, 0};
#endif

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    uint32x4_t vin0 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input0)), mask);
    uint32x4_t vin1 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input1)), mask);
    float32x4_t vout = vbslq_f32(vceqq_u32(vandq_u32(vin0, vin1), zeros), vfalse, vtrue);
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
  return NNACL_OK;
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

#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
  uint32x4_t mask = vmovq_n_u32(((uint32_t)(1u << 31) - 1));
  uint32x4_t zeros = {0, 0, 0, 0};
#endif

  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    uint32x4_t vin0 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input0)), mask);
    uint32x4_t vin1 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input1)), mask);
    float32x4_t vout = vbslq_f32(vceqq_u32(vorrq_u32(vin0, vin1), zeros), vfalse, vtrue);
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
  return NNACL_OK;
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
#ifdef ENABLE_NEON
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
  return NNACL_OK;
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
#ifdef ENABLE_NEON
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
  return NNACL_OK;
}

int BroadcastMinimum(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementMinimum(tile_input0, tile_input1, output, element_size);
}

float FloatNotEqualCheck(float in0, float in1) {
  float tmp = in0 - in1;
  if (tmp <= ACCURACY_DATA && tmp >= -ACCURACY_DATA) {
    return (float)false;
  }
  return (float)true;
}

int ElementNotEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vceqq_f32(vin0, vin1), vfalse, vtrue);
    vst1q_f32(output, vout);
#else
    output[0] = FloatNotEqualCheck(input0[0], input1[0]);
    output[1] = FloatNotEqualCheck(input0[1], input1[1]);
    output[2] = FloatNotEqualCheck(input0[2], input1[2]);
    output[3] = FloatNotEqualCheck(input0[3], input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] != input1[index]);
  }
  return NNACL_OK;
}

int BroadcastNotEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementNotEqual(tile_input0, tile_input1, output, element_size);
}

float FloatEqualCheck(float in0, float in1) {
  float tmp = in0 - in1;
  if (tmp <= ACCURACY_DATA && tmp >= -ACCURACY_DATA) {
    return (float)true;
  }
  return (float)false;
}

int ElementEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vceqq_f32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output, vout);
#else
    output[0] = FloatEqualCheck(input0[0], input1[0]);
    output[1] = FloatEqualCheck(input0[1], input1[1]);
    output[2] = FloatEqualCheck(input0[2], input1[2]);
    output[3] = FloatEqualCheck(input0[3], input1[3]);
#endif
    input0 += C4NUM;
    input1 += C4NUM;
    output += C4NUM;
  }
  for (int index = 0; index < block_mod; ++index) {
    output[index] = (float)(input0[index] == input1[index]);
  }
  return NNACL_OK;
}

int BroadcastEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                   int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementEqual(tile_input0, tile_input1, output, element_size);
}

int ElementLess(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcltq_f32(vin0, vin1), vtrue, vfalse);
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
  return NNACL_OK;
}

int BroadcastLess(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output, int element_size,
                  ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLess(tile_input0, tile_input1, output, element_size);
}

int ElementLessEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcleq_f32(vin0, vin1), vtrue, vfalse);
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
  return NNACL_OK;
}

int BroadcastLessEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                       int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLessEqual(tile_input0, tile_input1, output, element_size);
}

int ElementGreater(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcgtq_f32(vin0, vin1), vtrue, vfalse);
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
  return NNACL_OK;
}

int BroadcastGreater(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementGreater(tile_input0, tile_input1, output, element_size);
}

int ElementGreaterEqual(float *input0, float *input1, float *output, int element_size) {
  int block_mod = element_size % C4NUM;
  int block_c4 = element_size - block_mod;
#ifdef ENABLE_NEON
  float32x4_t vtrue = {1, 1, 1, 1};
  float32x4_t vfalse = {0, 0, 0, 0};
#endif
  for (int index = 0; index < block_c4; index += C4NUM) {
#ifdef ENABLE_NEON
    float32x4_t vin0 = vld1q_f32(input0);
    float32x4_t vin1 = vld1q_f32(input1);
    float32x4_t vout = vbslq_f32(vcgeq_f32(vin0, vin1), vtrue, vfalse);
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
  return NNACL_OK;
}

int BroadcastGreaterEqual(float *input0, float *input1, float *tile_input0, float *tile_input1, float *output,
                          int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementGreaterEqual(tile_input0, tile_input1, output, element_size);
}

#undef ACCURACY_DATA
