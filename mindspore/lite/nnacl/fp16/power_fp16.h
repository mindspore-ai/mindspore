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

#ifndef MINDSPORE_LITE_NNACL_FP16_POWER_FP16_H_
#define MINDSPORE_LITE_NNACL_FP16_POWER_FP16_H_

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/power_parameter.h"

#if defined(ENABLE_NEON)
typedef float16x8_t (*PowerSimdFunFp16)(float16x8_t x, const void *exponent);
#endif
typedef float16_t (*PowerScalarFunFp16)(float16_t x, const void *exponent);
typedef void (*PowerFunFp16)(const float16_t *, const float16_t *, float16_t *, int, float, float);

#ifdef __cplusplus
extern "C" {
#endif
static inline bool CheckInteger(float16_t f) { return floorf(f) == f; }

static inline float16_t StdPowerScalarFp16(float16_t x, const void *exponent) {
  return powf(x, *(float16_t *)exponent);
}

#if defined(ENABLE_NEON)
static inline float16x8_t StdPowerSimdFp16(float16x8_t x, const void *exponent) {
  float16x8_t result;
  result[0] = powf(x[0], *(float16_t *)exponent);
  result[1] = powf(x[1], *(float16_t *)exponent);
  result[2] = powf(x[2], *(float16_t *)exponent);
  result[3] = powf(x[3], *(float16_t *)exponent);
  result[4] = powf(x[4], *(float16_t *)exponent);
  result[5] = powf(x[5], *(float16_t *)exponent);
  result[6] = powf(x[6], *(float16_t *)exponent);
  result[7] = powf(x[7], *(float16_t *)exponent);
  return result;
}
#endif
int PowerFp16(const float16_t *input, const float16_t *exponent, float16_t *output, int len, float scale, float shift,
              bool broadcast);
void PowerSingleFp16(const float16_t *input, const float16_t *exponent, float16_t *output, int len, float scale,
                     float shift);
void PowerBroadCastFp16(const float16_t *input, const float16_t *exponent, float16_t *output, int len, float scale,
                        float shift);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP16_POWER_FP16_H_
