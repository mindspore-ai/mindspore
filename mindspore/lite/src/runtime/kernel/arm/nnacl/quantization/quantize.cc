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

#include "nnacl/quantization/quantize.h"

const uint64_t dSignMask = 1ull << 63;
const uint64_t dExponentMask = 0x7ffull << 52;
const uint64_t dFractionMask = (1ull << 52) - 1;
const int dExponentBias = 1022;
const int dMantissaBits = 52;
const int dInfiniteExponent = 0x7ff;
const double dNormalizer = 0x1p54;
const int dNormalizerBias = 54;
const int iMantissaBits = 31;

void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift) {
  if (quantized_multiplier == nullptr || shift == nullptr) {
    return;
  }
  // we split a floating number into two parts: exponent and fraction
  // since fraction is stored as int32, only 31 bits of mantissa is remained
  union {
    double d;
    uint64_t ul;
  } dul;
  dul.d = double_multiplier;
  if (!(dul.ul & (~dSignMask))) {
    // multiplier is 0
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  int exponent = (int) ((dul.ul & dExponentMask) >> dMantissaBits);
  if (exponent == dInfiniteExponent) {
    // multiplier is inf or NaN
    *shift = 0;
    if (!(dul.ul & dFractionMask)) {
      // inf
      *quantized_multiplier = (dul.ul & dSignMask) ? INT_MIN : INT_MAX;
    } else {
      // NaN
      *quantized_multiplier = 0;
    }
    return;
  }
  if (exponent == 0) {
    // multiplier is a subnormal number
    dul.d *= dNormalizer;
    exponent = (int) ((dul.ul & dExponentMask) >> dMantissaBits);
    *shift = exponent - dExponentBias - dNormalizerBias;
  } else {
    *shift = exponent - dExponentBias;
  }
  uint64_t fraction = dul.ul & dFractionMask;
  fraction += (1ull << dMantissaBits);
  uint64_t rounded = ((fraction >> (dMantissaBits - iMantissaBits)) + 1ull) >> 1;
  // we get 31 rounded bits now
  if (rounded == (1ull << iMantissaBits)) {
    // rounding may cause a carry
    rounded >>= 1;
    ++*shift;
  }
  *quantized_multiplier = (dul.ul & dSignMask) ? (-(int32_t)(rounded)) : (int32_t)(rounded);
}
