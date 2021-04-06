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

#include "nnacl/int8/quantize.h"

const uint64_t dSignMask = 1ull << 63;
const uint64_t dExponentMask = 0x7ffull << 52;
const uint64_t dFractionMask = (1ull << 52) - 1;
const int dExponentBias = 1022;
const int dMantissaBits = 52;
const int dInfiniteExponent = 0x7ff;
const double dNormalizer = 0x1p54;
const int dNormalizerBias = 54;
const int iMantissaBits = 31;

void QuantizeMultiplierSmallerThanOne(double double_multiplier, int32_t *quantized_multiplier, int *right_shift) {
  if (quantized_multiplier == NULL || right_shift == NULL) {
    return;
  }
  int shift = 0;
  QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  *right_shift = -shift;
}

void QuantizeRoundParameterWithDoublePrecision(double double_multiplier, int32_t *quantized_multiplier, int *left_shift,
                                               int *right_shift) {
  int shift = 0;
  QuantizeMultiplierSmallerThanOne(double_multiplier, quantized_multiplier, &shift);
  shift = -shift;
  if (shift < 0) {
    *left_shift = 0;
    *right_shift = shift;
  } else {
    *left_shift = shift;
    *right_shift = 0;
  }
}

void QuantizeRoundParameterWithSinglePrecision(double double_multiplier, int32_t *quantized_multiplier, int *left_shift,
                                               int *right_shift) {
  int shift = 0;
  const uint32_t scale_bits = (uint32_t)(double_multiplier);
  /* multiplier is in[0x40000000, 0x7FFFFF80] range */
  *quantized_multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  if (quantized_multiplier[0] < INT32_C(0x40000000) || quantized_multiplier[0] > INT32_C(0x7FFFFF80)) {
    return;
  }
  /* shift is in [0, 31] range */
  shift = 127 + 31 - 32 - ((uint32_t)(double_multiplier) >> 23);
  shift = -shift;
  if (shift < 0) {
    *left_shift = 0;
    *right_shift = shift;
  } else {
    *left_shift = shift;
    *right_shift = 0;
  }
}

uint8_t QuantizeToUint8(float real_value, float scale, int32_t zp) { return round(real_value / scale + zp); }

int32_t QuantizeToInt8(float real_value, float scale, int32_t zp) { return round(real_value / scale + zp); }

void CalculateActivationRangeQuantized(bool is_relu, bool is_relu6, int32_t zp, float scale, int *mini, int *maxi) {
  int32_t min = INT8_MIN;
  int32_t max = INT8_MAX;
  int32_t quantized_zero = QuantizeToInt8(0, scale, zp);
  int32_t quantized_six = QuantizeToInt8(6, scale, zp);
  if (is_relu) {
    min = min > quantized_zero ? min : quantized_zero;
  } else if (is_relu6) {
    min = min > quantized_zero ? min : quantized_zero;
    max = max < quantized_six ? max : quantized_six;
  } else {
    // do nothing
  }
  *mini = min;
  *maxi = max;
}

// quantize from float to int8
void Quantize(const float *input_data, int length, float scale, int zero_point, int8_t *output_data) {
  for (int i = 0; i < length; ++i) {
    int q = (int)round(input_data[i] / scale + zero_point);
    q = q > SCHAR_MAX ? SCHAR_MAX : q;
    q = q < SCHAR_MIN ? SCHAR_MIN : q;
    output_data[i] = (int8_t)q;
  }
}

// dequantize from int8 to float
void Dequantize(int8_t *input_data, int length, float scale, int zero_point, float *output_data) {
  for (int i = 0; i < length; ++i) {
    output_data[i] = scale * (input_data[i] - zero_point);
  }
}

void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift) {
  if (quantized_multiplier == NULL || shift == NULL) {
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
  int exponent = (int)((dul.ul & dExponentMask) >> dMantissaBits);
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
    exponent = (int)((dul.ul & dExponentMask) >> dMantissaBits);
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
