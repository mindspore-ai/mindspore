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

#include "nnacl/quantization/fixed_point.h"

// returns the high-32 bits of a * b with rounding
// assume that a and b is divided by 2^31, who fall into [-1, 1]
// so the mantissa of a * b is (a / 2^31) * (b / 2^31) * 2^31= (a * b) / 2^31
// actually we compute 2 * a * b / 2^32
// and take 32 bits of mantissa for rounding
int SaturatingRoundingDoublingHighMul(int a, int b) {
  if (a == INT_MIN && b == INT_MIN) {
    return INT_MAX;
  }
  int64_t ab = ((int64_t)a) * ((int64_t)b);
  int64_t rounding = ab >= 0 ? (1ll << 30) : (1ll - (1ll << 30));
  // do not apply right shift to potential negetive values
  int ab_mantissa = (int)((ab + rounding) / (1ll << 31));
  return ab_mantissa;
}

int16_t SaturatingRoundingDoublingHighMulInt16(int16_t a, int16_t b) {
  if (a == SHRT_MIN && b == SHRT_MIN) {
    return SHRT_MAX;
  }
  int32_t ab = ((int32_t)a) * ((int32_t)b);
  int16_t rounding = ab >= 0 ? (1ll << 14) : (1ll - (1ll << 14));
  return (int16_t)((ab + rounding) / (1ll << 15));
}

// division by a 2^exponent with rounding
// or arithmetic right shift with rouding
int RoundingDivideByPOT(int x, int exponent) {
  const int mask = (1ll << exponent) - 1;
  const int remainder = x & mask;
  const int threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

int MultiplyByQuantizedMultiplier(int32_t value, int32_t multiplier, int32_t left_shift, int32_t right_shift) {
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(value * (1 << left_shift), multiplier), -right_shift);
}

int FractionsBits(int kIntegerBits) {
  int totalBits = 8 * sizeof(int32_t) - 1;
  return totalBits - kIntegerBits;
}

int FixedPoint_One(int kIntegerBits, int kFractionsBits) {
  return (kIntegerBits == 0 ? INT32_MAX : ((1) << (uint32_t)(kIntegerBits == 0 ? 0 : kFractionsBits)));
}

int RoundingHalfSum(int a, int b) {
  int64_t a64 = a;
  int64_t b64 = b;
  int64_t sum = a64 + b64;
  int64_t sign = sum > 0 ? 1 : -1;
  return (int32_t)((sum + sign) / 2);
}

int32_t BitAnd(int32_t a, int32_t b) { return (uint32_t)a & (uint32_t)b; }

int32_t BitOr(int32_t a, int32_t b) { return (uint32_t)a | (uint32_t)b; }

int32_t BitXor(int32_t a, int32_t b) { return (uint32_t)a ^ (uint32_t)b; }

int32_t BitNot(int32_t a) { return ~(uint32_t)a; }

int SelectUsingMask(int mask, int bound, int val) { return BitXor(BitAnd(mask, bound), BitAnd(BitNot(mask), val)); }

int32_t MaskNonZero(int32_t a) {
  int32_t zreo = 0;
  return a ? BitNot(zreo) : zreo;
}

int SaturatingRoundingMultiplyByPOT(int32_t x, int Exponent) {
  int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0);
  if (ExponentSign == 0) {
    return x;
  } else if (ExponentSign == 1) {
    const int min = INT32_MIN;
    const int max = INT32_MAX;
    const int thresold = ((1 << (uint32_t)(31 - Exponent)) - 1);
    const int postive_mask = MaskNonZero(x > thresold);
    const int negative_mask = MaskNonZero(x < -thresold);
    int result = x << Exponent;
    result = SelectUsingMask(postive_mask, max, result);
    result = SelectUsingMask(negative_mask, min, result);
    return result;
  } else if (ExponentSign == -1) {
    return RoundingDivideByPOT(x, -Exponent);
  } else {
    return 0;
  }
}

int32_t Rescale(int x, int kIntegerBitsSrc, int kIntegerBitsDst) {
  int kExponent = kIntegerBitsSrc - kIntegerBitsDst;
  int result = SaturatingRoundingMultiplyByPOT(x, kExponent);
  return result;
}

static int32_t one_over_one_plus_x_for_x_in_0_1(int32_t a) {
  int one = FixedPoint_One(0, FractionsBits(0));
  int half_denominator = RoundingHalfSum(a, one);
  const int constant_48_over_17 = 1515870810;
  const int constant_neg_32_over_17 = -1010580540;
  int x = constant_48_over_17 + SaturatingRoundingDoublingHighMul(half_denominator, constant_neg_32_over_17);
  for (int i = 0; i < 3; i++) {
    int half_denominator_times_x = SaturatingRoundingDoublingHighMul(half_denominator, x);
    int one_minus_half_denominator_times_x = FixedPoint_One(2, FractionsBits(2)) - half_denominator_times_x;
    x = x + Rescale(SaturatingRoundingDoublingHighMul(x, one_minus_half_denominator_times_x), 2 + 2, 2);
  }
  return Rescale(x, 2 - 1, 0);
}

int CountLeadingZeroBits(uint32_t x) {
#if defined(__GUNC__)
  return x ? __builtin_clz(x) : 8 * sizeof(uint32_t);
#else
  if (x == 0) {
    return 8 * sizeof(uint32_t);
  }
  const int32_t leading_positive = (int32_t)(1) << (8 * sizeof(uint32_t) - 1);
  int leading_zeros = 0;
  while (x < leading_positive) {
    x <<= 1;
    leading_zeros++;
  }
  return leading_zeros;
#endif
}

int CountLeadingSignBits(int32_t x) {
#if defined(__GUNC__) && !defined(__clang__)
  return x ? __builtin_clrsb(x) : 8 * sizeof(int32_t);
#else
  return x >= 0 ? CountLeadingZeroBits((uint32_t)x) - 1 : x != INT32_MIN ? CountLeadingZeroBits(2 * (uint32_t)(-x)) : 0;
#endif
}

int32_t ComputerReciproal(int32_t x, int x_digits, int *recip_shift) {
  int leading_zreos_plus_one = CountLeadingZeroBits((uint32_t)x);
  *recip_shift = x_digits - leading_zreos_plus_one;
  const int32_t shifted_minus_one = (int32_t)(((uint32_t)x << leading_zreos_plus_one) - ((uint32_t)(1) << 31));
  const int32_t shifted_scaled = one_over_one_plus_x_for_x_in_0_1(shifted_minus_one);
  return shifted_scaled;
}
#ifdef ENABLE_NEON
int32x4_t RoundingDivideByPOTInt32x4(int32x4_t x, int exponent) {
  const int32x4_t shift_vec = vdupq_n_s32(-exponent);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
  const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
  return vrshlq_s32(fixed_up_x, shift_vec);
}

int32x4_t SaturatingRoundingDoublingHighMulInt32x4(int32x4_t a, int32x4_t b) { return vqrdmulhq_s32(a, b); }
#endif
