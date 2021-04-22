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

#include "nnacl/fp16/power_fp16.h"
#include "nnacl/errorcode.h"

#if defined(ENABLE_NEON)
float16x8_t OptimizedPowerSimdFp16(float16x8_t x, const void *exponent) {
  int tmp = (int)(*(float16_t *)exponent);
  int exp = abs(tmp);
  float16x8_t result = vmovq_n_f16(1.0f);
  while (exp) {
    if (exp % 2) {
      result *= x;
    }
    x *= x;
    exp = exp / 2;
  }
  if (tmp >= 0) {
    return result;
  }
  return 1 / result;
}
#endif

float16_t OptimizedPowerScalarFp16(float16_t x, const void *exponent) {
  int tmp = *(float16_t *)exponent;
  int exp = abs(tmp);
  float16_t result = 1;
  while (exp) {
    if (exp % 2) {
      result *= x;
    }
    x *= x;
    exp = exp / 2;
  }
  return tmp >= 0 ? result : 1 / result;
}

void PowerBroadCastFp16(const float16_t *input, const float16_t *exponent, float16_t *output, int len, float scale,
                        float shift) {
  PowerScalarFunFp16 PowerScalarFunFp16_ = NULL;
#if defined(ENABLE_NEON)
  PowerSimdFunFp16 PowerSimdFunFp16_ = NULL;
#endif

  if (CheckInteger(*exponent)) {
#if defined(ENABLE_NEON)
    PowerSimdFunFp16_ = OptimizedPowerSimdFp16;
#endif
    PowerScalarFunFp16_ = OptimizedPowerScalarFp16;
  } else {
#if defined(ENABLE_NEON)
    PowerSimdFunFp16_ = StdPowerSimdFp16;
#endif
    PowerScalarFunFp16_ = StdPowerScalarFp16;
  }
  int i = 0;
#ifdef ENABLE_NEON
  int len_c8 = UP_ROUND(len, C8NUM);
  float16x8_t scale_8 = vmovq_n_f16(scale);
  float16x8_t shift_8 = vmovq_n_f16(shift);
  for (; i < len_c8; i += C8NUM) {
    float16x8_t result = PowerSimdFunFp16_(scale_8 * vld1q_f16(input + i) + shift_8, exponent);
    vst1q_f16(output + i, result);
  }
#endif
  for (; i < len; ++i) {
    output[i] = PowerScalarFunFp16_(scale * input[i] + shift, exponent);
  }
}

void PowerSingleFp16(const float16_t *input, const float16_t *exponent, float16_t *output, int len, float scale,
                     float shift) {
  int i = 0;
  PowerScalarFunFp16 PowerScalarFunFp16_ = NULL;
#ifdef ENABLE_NEON
  int len_c8 = UP_ROUND(len, C8NUM);
  float16x8_t scale_8 = vmovq_n_f16(scale);
  float16x8_t shift_8 = vmovq_n_f16(shift);
  for (; i < len_c8; i += C8NUM) {
    float16x8_t tmp_8 = scale_8 * vld1q_f16(input + i) + shift_8;
    for (int j = 0; j < 8; ++j) {
      PowerScalarFunFp16_ = CheckInteger(exponent[i + j]) ? OptimizedPowerScalarFp16 : StdPowerScalarFp16;
      output[i + j] = PowerScalarFunFp16_(tmp_8[j], exponent + i + j);
    }
  }
#endif
  for (; i < len; ++i) {
    PowerScalarFunFp16_ = CheckInteger(exponent[i]) ? OptimizedPowerScalarFp16 : StdPowerScalarFp16;
    output[i] = PowerScalarFunFp16_(scale * input[i] + shift, exponent + i);
  }
}

int PowerFp16(const float16_t *input, const float16_t *exponent, float16_t *output, int len, float scale, float shift,
              bool broadcast) {
  if (input == NULL || exponent == NULL || output == NULL) {
    return NNACL_NULL_PTR;
  }
  PowerFunFp16 PowerFunFp16_ = NULL;
  PowerFunFp16_ = broadcast ? PowerBroadCastFp16 : PowerSingleFp16;
  PowerFunFp16_(input, exponent, output, len, scale, shift);
  return NNACL_OK;
}
