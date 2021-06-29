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
#ifndef MINDSPORE_NNACL_CAST_FP16_H_
#define MINDSPORE_NNACL_CAST_FP16_H_

#include "nnacl/op_base.h"
#if defined(ENABLE_ARM) && defined(ENABLE_FP16)
#include <arm_neon.h>

#ifdef __cplusplus
extern "C" {
#endif

inline void BoolToFloat16(const bool *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

inline void Uint8ToFloat16(const uint8_t *input, float16_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

inline void Float16ToInt32(const float16_t *input, int32_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int32_t)input[i];
  }
}

inline void Float16ToInt64(const float16_t *input, int64_t *output, int number) {
  for (int i = 0; i < number; ++i) {
    output[i] = (int64_t)input[i];
  }
}

#ifdef ENABLE_ARM64
inline void Float32ToFloat16(const float *__restrict input, float16_t *__restrict output, int number) {
  int count = (number & ~(C8NUM - 1));
  int i = 0;
  for (; i < count; i += C8NUM) {
    float32x4_t in1 = vld1q_f32(input + i);
    float16x4_t out1 = vcvt_f16_f32(in1);
    float32x4_t in2 = vld1q_f32(input + i + 4);
    float16x4_t out2 = vcvt_f16_f32(in2);
    float16x8_t out = vcombine_f16(out1, out2);
    vst1q_f16(output + i, out);
  }
  for (; i < number; ++i) {
    output[i] = (float16_t)input[i];
  }
}

inline void Float16ToFloat32(const float16_t *__restrict input, float *__restrict output, int number) {
  int count = number & ~(C8NUM - 1);
  int i = 0;
  for (; i < count; i += C8NUM) {
    float16x8_t in = vld1q_f16(input + i);
    float16x4_t in1 = vget_low_f16(in);
    float16x4_t in2 = vget_high_f16(in);
    float32x4_t out1 = vcvt_f32_f16(in1);
    vst1q_f32(output + i, out1);
    float32x4_t out2 = vcvt_f32_f16(in2);
    vst1q_f32(output + i + 4, out2);
  }
  for (; i < number; ++i) {
    output[i] = (float)input[i];
  }
}
#else
void Float32ToFloat16(const float *input, float16_t *output, int number);

void Float16ToFloat32(const float16_t *input, float *output, int number);
#endif

#ifdef __cplusplus
}
#endif
#endif
#endif  // MINDSPORE_NNACL_CAST_FP16_H_
