/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/power_fp32.h"
#include "nnacl/errorcode.h"

#if defined(ENABLE_ARM) || defined(ENABLE_AVX) || defined(ENABLE_SSE)
MS_FLOAT32X4 OptimizedPowerSimd(MS_FLOAT32X4 x, const void *exponent) {
  int exp = abs((int)(*(float *)exponent));
  MS_FLOAT32X4 result = MS_MOVQ_F32(1.0f);
  while (exp) {
    if (exp % 2) {
      result *= x;
    }
    x *= x;
    exp = exp / 2;
  }
  if (*(float *)exponent >= 0) {
    return result;
  }
  return 1 / result;
}
#endif

float OptimizedPowerScalar(float x, const void *exponent) {
  int exp = abs((int)(*(float *)exponent));
  float result = 1;
  while (exp) {
    if (exp % 2) {
      result *= x;
    }
    x *= x;
    exp = exp / 2;
  }
  return *(float *)exponent >= 0 ? result : 1 / result;
}

void PowerBroadCast(const float *input, const float *exponent, float *output, int len, float scale, float shift) {
  PowerScalarFun PowerScalarFun_ = NULL;
#if defined(ENABLE_ARM) || defined(ENABLE_AVX) || defined(ENABLE_SSE)
  PowerSimdFun PowerSimdFun_ = NULL;
#endif

  if (CheckInteger(*exponent)) {
#if defined(ENABLE_ARM) || defined(ENABLE_AVX) || defined(ENABLE_SSE)
    PowerSimdFun_ = OptimizedPowerSimd;
#endif
    PowerScalarFun_ = OptimizedPowerScalar;
  } else {
#if defined(ENABLE_ARM) || defined(ENABLE_AVX) || defined(ENABLE_SSE)
    PowerSimdFun_ = StdPowerSimd;
#endif
    PowerScalarFun_ = StdPowerScalar;
  }
  int i = 0;
#if defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM)
  int len_c4 = UP_ROUND(len, C4NUM);
  MS_FLOAT32X4 scale_4 = MS_MOVQ_F32(scale);
  MS_FLOAT32X4 shift_4 = MS_MOVQ_F32(shift);
  for (; i < len_c4; i += C4NUM) {
    MS_FLOAT32X4 result = PowerSimdFun_(scale_4 * MS_LDQ_F32(input + i) + shift_4, exponent);
    MS_STQ_F32(output + i, result);
  }
#endif
  for (; i < len; ++i) {
    output[i] = PowerScalarFun_(scale * input[i] + shift, exponent);
  }
}

void PowerSingle(const float *input, const float *exponent, float *output, int len, float scale, float shift) {
  int i = 0;
  PowerScalarFun PowerScalarFun_ = NULL;
#if defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM)
  int len_c4 = UP_ROUND(len, C4NUM);
  MS_FLOAT32X4 scale_4 = MS_MOVQ_F32(scale);
  MS_FLOAT32X4 shift_4 = MS_MOVQ_F32(shift);
  for (; i < len_c4; i += C4NUM) {
    MS_FLOAT32X4 tmp_4 = scale_4 * MS_LDQ_F32(input + i) + shift_4;
    for (int j = 0; j < 4; ++j) {
      PowerScalarFun_ = CheckInteger(exponent[i + j]) ? OptimizedPowerScalar : StdPowerScalar;
      output[i + j] = PowerScalarFun_(tmp_4[j], exponent + i + j);
    }
  }
#endif
  for (; i < len; ++i) {
    PowerScalarFun_ = CheckInteger(exponent[i]) ? OptimizedPowerScalar : StdPowerScalar;
    output[i] = PowerScalarFun_(scale * input[i] + shift, exponent + i);
  }
}

int Power(const float *input, const float *exponent, float *output, int len, float scale, float shift, bool broadcast) {
  if (input == NULL || exponent == NULL || output == NULL) {
    return NNACL_NULL_PTR;
  }
  PowerFun PowerFun_ = NULL;
  PowerFun_ = broadcast ? PowerBroadCast : PowerSingle;
  PowerFun_(input, exponent, output, len, scale, shift);
  return NNACL_OK;
}
