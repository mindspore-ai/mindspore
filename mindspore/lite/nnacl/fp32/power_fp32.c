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

#include "nnacl/fp32/power_fp32.h"

bool CheckInteger(float f) { return floorf(f) == f; }

float OptimizedPowerImpl(float x, int exponent) {
  int exp = abs(exponent);
  float result = 1;
  float iterator = x;
  while (exp) {
    if (exp % 2) {
      result *= iterator;
    }
    iterator *= iterator;
    exp = exp / 2;
  }
  return exponent >= 0 ? result : 1 / result;
}

float StdPowerImpl(float x, float exponent) { return pow(x, exponent); }

void Power(const float *input, const float *exponent, float *output, int len, float scale, float shift,
           bool broadcast) {
  if (input == NULL || exponent == NULL) {
    return;
  }
  if (broadcast) {
    if (CheckInteger(*exponent)) {
      for (int i = 0; i < len; ++i) {
        output[i] = OptimizedPowerImpl(scale * input[i] + shift, (int)(*exponent));
      }
    } else {
      for (int i = 0; i < len; ++i) {
        output[i] = StdPowerImpl(scale * input[i] + shift, *exponent);
      }
    }
  } else {
    for (int i = 0; i < len; ++i) {
      if (CheckInteger(*exponent)) {
        output[i] = OptimizedPowerImpl(scale * input[i] + shift, (int)exponent[i]);
      } else {
        output[i] = StdPowerImpl(scale * input[i] + shift, exponent[i]);
      }
    }
  }
}
