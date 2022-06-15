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
#include "nnacl/power_fp32_simd.h"

float OptimizedPowerScalar(float x, const float *exponent) {
  int exp = abs((int)(*exponent));
  float result = 1;
  while (exp) {
    if (exp % 2) {
      result *= x;
    }
    x *= x;
    exp = exp / 2;
  }
  return *exponent >= 0 ? result : 1 / result;
}

void PowerBroadCast(const float *input, const float *exponent, float *output, int len, float scale, float shift) {
  PowerScalarFun PowerScalarFun_ = NULL;

  int i = 0;
  if (CheckInteger(*exponent)) {
    PowerScalarFun_ = OptimizedPowerScalar;
    SIMD_RUN_NO_SCALAR(PowerBroadCastIntExponent, i, input, (int)(*exponent), output, len, scale, shift);
  } else {
    PowerScalarFun_ = StdPowerScalar;
    SIMD_RUN_NO_SCALAR(PowerBroadCastFloatExponent, i, input, *exponent, output, len, scale, shift);
  }

  for (; i < len; ++i) {
    output[i] = PowerScalarFun_(scale * input[i] + shift, exponent);
  }
}

void PowerSingle(const float *input, const float *exponent, float *output, int len, float scale, float shift) {
  int i = 0;

  SIMD_RUN_NO_SCALAR(PowerSingleExponent, i, input, exponent, output, len, scale, shift);
  PowerScalarFun PowerScalarFun_ = NULL;
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
