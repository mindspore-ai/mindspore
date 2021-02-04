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

#include "nnacl/fp32/arithmetic_fp32.h"
#include <math.h>
#include <float.h>

#define ACCURACY_DATA 0.00000001

int ElementOptMul(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vmulq_f32(vin0_opt, vin1);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] * input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vmulq_f32(vin0, vin1_opt);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] * input1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptMulRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vmaxq_f32(vmulq_f32(vin0_opt, vin1), zeros);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[0] * input1[index], 0);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vmaxq_f32(vmulq_f32(vin0, vin1_opt), zeros);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[index] * input1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptMulRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vmulq_f32(vin0_opt, vin1), zeros), bounds);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] * input1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vmulq_f32(vin0, vin1_opt), zeros), bounds);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] * input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementOptMulInt(const int *input0, const int *input1, int *output, const int element_size,
                     const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  int32x4_t vin0_opt = vdupq_n_s32(input0[0]);
  int32x4_t vin1_opt = vdupq_n_s32(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin1 = vld1q_s32(input1 + index);
      int32x4_t vout = vmulq_s32(vin0_opt, vin1);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] * input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin0 = vld1q_s32(input0 + index);
      int32x4_t vout = vmulq_s32(vin0, vin1_opt);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] * input1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptMulReluInt(const int *input0, const int *input1, int *output, const int element_size,
                         const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  int32x4_t vin0_opt = vdupq_n_s32(input0[0]);
  int32x4_t vin1_opt = vdupq_n_s32(input1[0]);
  int32x4_t zeros = vdupq_n_s32(0);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin1 = vld1q_s32(input1 + index);
      int32x4_t vout = vmaxq_s32(vmulq_s32(vin0_opt, vin1), zeros);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[0] * input1[index], 0);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin0 = vld1q_s32(input0 + index);
      int32x4_t vout = vmaxq_s32(vmulq_s32(vin0, vin1_opt), zeros);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[index] * input1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptMulRelu6Int(const int *input0, const int *input1, int *output, const int element_size,
                          const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  int32x4_t vin0_opt = vdupq_n_s32(input0[0]);
  int32x4_t vin1_opt = vdupq_n_s32(input1[0]);
  int32x4_t zeros = vdupq_n_s32(0);
  int32x4_t bounds = vdupq_n_s32(6);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin1 = vld1q_s32(input1 + index);
      int32x4_t vout = vminq_s32(vmaxq_s32(vmulq_s32(vin0_opt, vin1), zeros), bounds);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] * input1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin0 = vld1q_s32(input0 + index);
      int32x4_t vout = vminq_s32(vmaxq_s32(vmulq_s32(vin0, vin1_opt), zeros), bounds);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] * input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementOptSub(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vsubq_f32(vin0_opt, vin1);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] - input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vsubq_f32(vin0, vin1_opt);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] - input1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptSubRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vmaxq_f32(vsubq_f32(vin0_opt, vin1), zeros);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[0] - input1[index], 0);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vmaxq_f32(vsubq_f32(vin0, vin1_opt), zeros);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[index] - input1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptSubRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vsubq_f32(vin0_opt, vin1), zeros), bounds);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] - input1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vsubq_f32(vin0, vin1_opt), zeros), bounds);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] - input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementOptAdd(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vaddq_f32(vin0_opt, vin1);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] + input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vaddq_f32(vin0, vin1_opt);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] + input1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptAddInt(const int *input0, const int *input1, int *output, const int element_size,
                     const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  int32x4_t vin0_opt = vdupq_n_s32(input0[0]);
  int32x4_t vin1_opt = vdupq_n_s32(input1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin1 = vld1q_s32(input1 + index);
      int32x4_t vout = vaddq_s32(vin0_opt, vin1);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[0] + input1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      int32x4_t vin0 = vld1q_s32(input0 + index);
      int32x4_t vout = vaddq_s32(vin0, vin1_opt);
      vst1q_s32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = input0[index] + input1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptAddRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vmaxq_f32(vaddq_f32(vin0_opt, vin1), zeros);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[0] + input1[index], 0);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vmaxq_f32(vaddq_f32(vin0, vin1_opt), zeros);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMAX(input0[index] + input1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptAddRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(input0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(input1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(input1 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vaddq_f32(vin0_opt, vin1), zeros), bounds);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] + input1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= element_size - 4; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(input0 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vaddq_f32(vin0, vin1_opt), zeros), bounds);
      vst1q_f32(output + index, vout);
    }
#endif
    for (; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] + input1[0], 0), 6);
    }
  }

  return NNACL_OK;
}

int ElementOptDiv(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < element_size; index++) {
      output[index] = input0[0] / input1[index];
    }
  } else {
    if (input1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    for (int index = 0; index < element_size; index++) {
      output[index] = input0[index] / input1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptDivRelu(const float *input0, const float *input1, float *output, const int element_size,
                      const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < element_size; index++) {
      output[index] = input0[0] / input1[index];
      output[index] = output[index] > 0 ? output[index] : 0;
    }
  } else {
    for (int index = 0; index < element_size; index++) {
      output[index] = input0[index] / input1[0];
      output[index] = output[index] > 0 ? output[index] : 0;
    }
  }
  return NNACL_OK;
}

int ElementOptDivRelu6(const float *input0, const float *input1, float *output, const int element_size,
                       const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[0] / input1[index], 0), 6);
    }
  } else {
    for (int index = 0; index < element_size; index++) {
      output[index] = MSMIN(MSMAX(input0[index] / input1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementOptDivInt(const int *input0, const int *input1, int *output, const int element_size,
                     const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < element_size; index++) {
      output[index] = input0[0] / input1[index];
    }
  } else {
    if (input1[0] == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }
    for (int index = 0; index < element_size; index++) {
      output[index] = input0[index] / input1[0];
    }
  }
  return NNACL_OK;
}

int ElementMul(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vmulq_f32(vin0, vin1);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] * input1[index];
  }
  return NNACL_OK;
}

int ElementMulRelu(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vmulq_f32(vin0, vin1);
    vout = vbslq_f32(vcgtq_f32(vout, zeros), vout, zeros);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    float res = input0[index] * input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementMulRelu6(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vminq_f32(vmaxq_f32(vmulq_f32(vin0, vin1), zeros), bounds);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(MSMAX(input0[index] * input1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementMulInt(const int *input0, const int *input1, int *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    int32x4_t vin0 = vld1q_s32(input0 + index);
    int32x4_t vin1 = vld1q_s32(input1 + index);
    int32x4_t vout = vmulq_s32(vin0, vin1);
    vst1q_s32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] * input1[index];
  }
  return NNACL_OK;
}

int ElementMulReluInt(const int *input0, const int *input1, int *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  int32x4_t zeros = vdupq_n_s32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    int32x4_t vin0 = vld1q_s32(input0 + index);
    int32x4_t vin1 = vld1q_s32(input1 + index);
    int32x4_t vout = vmulq_s32(vin0, vin1);
    vout = vbslq_s32(vcgtq_s32(vout, zeros), vout, zeros);
    vst1q_s32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    float res = input0[index] * input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementMulRelu6Int(const int *input0, const int *input1, int *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  int32x4_t zeros = vdupq_n_s32(0);
  int32x4_t bounds = vdupq_n_s32(6);
  for (; index <= element_size - 4; index += C4NUM) {
    int32x4_t vin0 = vld1q_s32(input0 + index);
    int32x4_t vin1 = vld1q_s32(input1 + index);
    int32x4_t vout = vminq_s32(vmaxq_s32(vmulq_s32(vin0, vin1), zeros), bounds);
    vst1q_s32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(MSMAX(input0[index] * input1[index], 0), 6);
  }
  return NNACL_OK;
}

int BroadcastMul(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                 int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementMul(tile_input0, tile_input1, output, element_size);
}

int ElementAdd(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vaddq_f32(vin0, vin1);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] + input1[index];
  }
  return NNACL_OK;
}

int ElementAddRelu(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vaddq_f32(vin0, vin1);
    vout = vbslq_f32(vcgtq_f32(vout, zeros), vout, zeros);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    float res = input0[index] + input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementAddRelu6(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vminq_f32(vmaxq_f32(vaddq_f32(vin0, vin1), zeros), bounds);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(MSMAX(input0[index] + input1[index], 0), 6);
  }
  return NNACL_OK;
}

int ElementAddInt(const int *input0, const int *input1, int *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    int32x4_t vin0 = vld1q_s32(input0 + index);
    int32x4_t vin1 = vld1q_s32(input1 + index);
    int32x4_t vout = vaddq_s32(vin0, vin1);
    vst1q_s32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] + input1[index];
  }
  return NNACL_OK;
}

int ElementAddInt8(const int8_t *input0, const int8_t *input1, int8_t *output, int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] + input1[i];
  }
  return NNACL_OK;
}

int BroadcastAdd(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                 int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementAdd(tile_input0, tile_input1, output, element_size);
}

int BroadcastAddInt8(const int8_t *input0, const int8_t *input1, int8_t *tile_input0, int8_t *tile_input1,
                     int8_t *output, int element_size, ArithmeticParameter *param) {
  TileDimensionsInt8(input0, input1, tile_input0, tile_input1, param);
  return ElementAddInt8(tile_input0, tile_input1, output, element_size);
}

int ElementSub(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vsubq_f32(vin0, vin1);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] - input1[index];
  }
  return NNACL_OK;
}

int ElementSubInt(const int *input0, const int *input1, int *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    int32x4_t vin0 = vld1q_s32(input0 + index);
    int32x4_t vin1 = vld1q_s32(input1 + index);
    int32x4_t vout = vsubq_s32(vin0, vin1);
    vst1q_s32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] - input1[index];
  }
  return NNACL_OK;
}

int ElementSubRelu(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vsubq_f32(vin0, vin1);
    vout = vbslq_f32(vcgtq_f32(vout, zeros), vout, zeros);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    float res = input0[index] - input1[index];
    output[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementSubRelu6(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vminq_f32(vmaxq_f32(vsubq_f32(vin0, vin1), zeros), bounds);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = MSMIN(MSMAX(input0[index] - input1[index], 0), 6);
  }

  return NNACL_OK;
}

int BroadcastSub(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                 int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementSub(tile_input0, tile_input1, output, element_size);
}

int ElementDiv(const float *input0, const float *input1, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] / input1[i];
  }
  return NNACL_OK;
}

int ElementDivRelu(const float *input0, const float *input1, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    float res = input0[i] / input1[i];
    output[i] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementDivRelu6(const float *input0, const float *input1, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = MSMIN(MSMAX(input0[i] / input1[i], 0), 6);
  }
  return NNACL_OK;
}

int BroadcastDiv(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                 int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementDiv(tile_input0, tile_input1, output, element_size);
}

int ElementFloorMod(const float *input0, const float *input1, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] - floorf(input0[i] / input1[i]) * input1[i];
  }
  return NNACL_OK;
}

int ElementFloorModInt(const int *input0, const int *input1, int *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = input0[i] - (int)floor((double)input0[i] / (double)input1[i]) * input1[i];
  }
  return NNACL_OK;
}

int BroadcastFloorMod(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementFloorMod(tile_input0, tile_input1, output, element_size);
}

int ElementMod(const float *input0, const float *input1, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = fmod(input0[i], input1[i]);
  }
  return NNACL_OK;
}

int ElementModInt(const int *input0, const int *input1, int *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = fmod(input0[i], input1[i]);
  }
  return NNACL_OK;
}

int ElementOptMod(const float *input0, const float *input1, float *output, const int element_size,
                  const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < element_size; index++) {
      output[index] = fmod(input0[0], input1[index]);
    }
  } else {
    for (int index = 0; index < element_size; index++) {
      output[index] = fmod(input0[index], input1[0]);
    }
  }
  return NNACL_OK;
}

int ElementOptModInt(const int *input0, const int *input1, int *output, const int element_size,
                     const ArithmeticParameter *param) {
  if (param->in_elements_num0_ == 1) {
    for (int index = 0; index < element_size; index++) {
      output[index] = fmod(input0[0], input1[index]);
    }
  } else {
    for (int index = 0; index < element_size; index++) {
      output[index] = fmod(input0[index], input1[0]);
    }
  }
  return NNACL_OK;
}

int ElementFloorDiv(const float *input0, const float *input1, float *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = floorf(input0[i] / input1[i]);
  }
  return NNACL_OK;
}

int ElementFloorDivInt(const int *input0, const int *input1, int *output, const int element_size) {
  for (int i = 0; i < element_size; i++) {
    output[i] = (int)floor((double)input0[i] / (double)input1[i]);
  }
  return NNACL_OK;
}

int BroadcastFloorDiv(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                      int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementFloorDiv(tile_input0, tile_input1, output, element_size);
}

int ElementLogicalAnd(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  uint32x4_t mask = vmovq_n_u32(((uint32_t)(1u << 31) - 1));
  uint32x4_t zeros = vdupq_n_u32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    uint32x4_t vin0 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input0 + index)), mask);
    uint32x4_t vin1 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input1 + index)), mask);
    float32x4_t vout = vbslq_f32(vceqq_u32(vandq_u32(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)((bool)(input0[index]) & (bool)(input1[index]));
  }
  return NNACL_OK;
}

int ElementLogicalAndInt(const int *input0, const int *input1, int *output, const int element_size) {
  int index = 0;
  for (; index < element_size; index++) {
    output[index] = (int)((int)(input0[index]) & (int)(input1[index]));
  }
  return NNACL_OK;
}

int ElementLogicalAndBool(const bool *input0, const bool *input1, bool *output, const int element_size) {
  int index = 0;
  for (; index < element_size; index++) {
    output[index] = (bool)((bool)(input0[index]) & (bool)(input1[index]));
  }
  return NNACL_OK;
}

int ElementSquaredDifference(const float *input0, const float *input1, float *output, const int element_size) {
  ElementSub(input0, input1, output, element_size);
  return ElementMul(output, output, output, element_size);
}

int BroadcastSquaredDifference(const float *input0, const float *input1, float *tile_input0, float *tile_input1,
                               float *output, int element_size, ArithmeticParameter *param) {
  BroadcastSub(input0, input1, tile_input0, tile_input1, output, element_size, param);
  return ElementMul(output, output, output, element_size);
}

int BroadcastLogicalAnd(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                        int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLogicalAnd(tile_input0, tile_input1, output, element_size);
}

int ElementLogicalOr(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  uint32x4_t mask = vmovq_n_u32(((uint32_t)(1u << 31) - 1));
  uint32x4_t zeros = vdupq_n_u32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    uint32x4_t vin0 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input0 + index)), mask);
    uint32x4_t vin1 = vandq_u32(vreinterpretq_s32_f32(vld1q_f32(input1 + index)), mask);
    float32x4_t vout = vbslq_f32(vceqq_u32(vorrq_u32(vin0, vin1), zeros), vfalse, vtrue);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)((bool)(input0[index]) | (bool)(input1[index]));
  }
  return NNACL_OK;
}

int BroadcastLogicalOr(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                       int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLogicalOr(tile_input0, tile_input1, output, element_size);
}

int ElementMaximum(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vmaxq_f32(vin0, vin1);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] > input1[index] ? input0[index] : input1[index];
  }
  return NNACL_OK;
}

int BroadcastMaximum(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementMaximum(tile_input0, tile_input1, output, element_size);
}

int ElementMinimum(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vminq_f32(vin0, vin1);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = input0[index] > input1[index] ? input1[index] : input0[index];
  }
  return NNACL_OK;
}

int BroadcastMinimum(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
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

int ElementNotEqual(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vbslq_f32(vceqq_f32(vin0, vin1), vfalse, vtrue);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)(fabsf(input0[index] - input1[index]) > FLT_EPSILON);
  }
  return NNACL_OK;
}

int BroadcastNotEqual(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
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

int ElementEqual(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vbslq_f32(vceqq_f32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)(fabsf(input0[index] - input1[index]) <= FLT_EPSILON);
  }
  return NNACL_OK;
}

int BroadcastEqual(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                   int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementEqual(tile_input0, tile_input1, output, element_size);
}

int ElementLess(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vbslq_f32(vcltq_f32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)(input0[index] < input1[index]);
  }
  return NNACL_OK;
}

int BroadcastLess(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                  int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLess(tile_input0, tile_input1, output, element_size);
}

int ElementLessEqual(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vbslq_f32(vcleq_f32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)(input0[index] <= input1[index]);
  }
  return NNACL_OK;
}

int BroadcastLessEqual(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                       int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementLessEqual(tile_input0, tile_input1, output, element_size);
}

int ElementGreater(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vbslq_f32(vcgtq_f32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)(input0[index] > input1[index]);
  }
  return NNACL_OK;
}

int BroadcastGreater(const float *input0, const float *input1, float *tile_input0, float *tile_input1, float *output,
                     int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementGreater(tile_input0, tile_input1, output, element_size);
}

int ElementGreaterEqual(const float *input0, const float *input1, float *output, const int element_size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t vtrue = vdupq_n_f32(1);
  float32x4_t vfalse = vdupq_n_f32(0);
  for (; index <= element_size - 4; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(input0 + index);
    float32x4_t vin1 = vld1q_f32(input1 + index);
    float32x4_t vout = vbslq_f32(vcgeq_f32(vin0, vin1), vtrue, vfalse);
    vst1q_f32(output + index, vout);
  }
#endif
  for (; index < element_size; index++) {
    output[index] = (float)(input0[index] >= input1[index]);
  }
  return NNACL_OK;
}

int BroadcastGreaterEqual(const float *input0, const float *input1, float *tile_input0, float *tile_input1,
                          float *output, int element_size, ArithmeticParameter *param) {
  TileDimensions(input0, input1, tile_input0, tile_input1, param);
  return ElementGreaterEqual(tile_input0, tile_input1, output, element_size);
}

#undef ACCURACY_DATA

#ifdef ENABLE_NNACL_INFER_SHAPE
int ArithmeticInferShape(int **in_shape, size_t *dim_size, int *out_shape, int *in_format, int *out_format,
                         int *in_datatype, int *out_datatype, OpParameter *param) {
  *out_format = in_format[0];
  *out_datatype = in_datatype[0];
  const ArithmeticParameter *arithmetic_parameter = (const ArithmeticParameter *)param;
  int ndim0 = dim_size[0];
  int ndim1 = dim_size[1];
  int *in_shape0 = in_shape[0];
  int *in_shape1 = in_shape[1];
  if (ndim0 < ndim1) {
    arithmetic_parameter->ndim_ = ndim1;
    int fill_dim_num = ndim1 - ndim0;
    int j = 0;
    for (int i = 0; i < ndim1; ++i) {
      if (i < fill_dim_num) {
        arithmetic_parameter->in_shape0_[i] = 1;
      } else {
        arithmetic_parameter->in_shape0_[i] = in_shape0[j++];
      }
      arithmetic_parameter->in_shape1_[i] = in_shape1[i];
    }
  } else if (ndim0 > ndim1) {
    arithmetic_parameter->ndim_ = ndim0;
    int fill_dim_num = ndim0 - ndim1;
    int j = 0;
    for (int i = 0; i < ndim0; ++i) {
      if (i < fill_dim_num) {
        arithmetic_parameter->in_shape1_[i] = 1;
      } else {
        arithmetic_parameter->in_shape1_[i] = in_shape1[j++];
      }
      arithmetic_parameter->in_shape0_[i] = in_shape0[i];
    }
  } else {
    arithmetic_parameter->ndim_ = ndim0;
    for (int i = 0; i < ndim0; ++i) {
      arithmetic_parameter->in_shape0_[i] = in_shape0[i];
      arithmetic_parameter->in_shape1_[i] = in_shape1[i];
    }
  }
  int j = 0;
  for (size_t i = 0; i < arithmetic_parameter->ndim_; ++i) {
    if (arithmetic_parameter->in_shape0_[i] != arithmetic_parameter->in_shape1_[i]) {
      if (arithmetic_parameter->in_shape0_[i] == 1) {
        out_shape[j++] = arithmetic_parameter->in_shape1_[i];
      } else if (arithmetic_parameter->in_shape1_[i] == 1) {
        out_shape[j++] = arithmetic_parameter->in_shape0_[i];
      } else {
        return NNACL_PARAM_INVALID;
      }
    } else {
      out_shape[j++] = arithmetic_parameter->in_shape0_[i];
    }
  }
  return NNACL_OK;
}
#endif
