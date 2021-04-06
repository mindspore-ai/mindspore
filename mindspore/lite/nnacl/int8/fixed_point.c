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

#include "nnacl/int8/fixed_point.h"

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
// or arithmetic right shift with rounding
int RoundingDivideByPOT(int x, int exponent) {
  const int mask = (1ll << exponent) - 1;
  const int remainder = x & mask;
  const int threshold = (mask >> 1) + (x < 0 ? 1 : 0);
  return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

int UpwardRounding(int x, int exponent) {
  const int32_t rounding_offset = (exponent > 0) ? (1 << (exponent - 1)) : 0;
  if (x > INT32_MAX - rounding_offset) {
    return 1 << (31 - exponent);
  }
  return (x + rounding_offset) >> exponent;
}

int MultiplyByQuantizedMultiplier(int32_t value, int32_t multiplier, int32_t left_shift, int32_t right_shift) {
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(value * (1 << left_shift), multiplier), -right_shift);
}

int MultiplyByQuantizedMultiplierWithUpwardRounding(int32_t value, int32_t multiplier, int32_t left_shift,
                                                    int32_t right_shift) {
  return UpwardRounding(SaturatingRoundingDoublingHighMul(value * (1 << left_shift), multiplier), -right_shift);
}

int MultiplyByMultiplierAndRightShift(int32_t value, int32_t multiplier, int32_t right_shift) {
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(value, multiplier), right_shift);
}

int FractionsBits(int integer_bits) { return 8 * sizeof(int32_t) - 1 - integer_bits; }

int FixedPoint_One(int integer_bits, int fractions_bits) {
  return (integer_bits == 0 ? INT32_MAX : ((1) << (uint32_t)(integer_bits == 0 ? 0 : fractions_bits)));
}

int RoundingHalfSum(int32_t a, int32_t b) {
  int64_t sum = (int64_t)a + (int64_t)b;
  return (int32_t)((sum + (sum > 0 ? 1 : -1)) / 2);
}

int32_t BitAnd(int32_t a, int32_t b) { return (uint32_t)a & (uint32_t)b; }

int32_t BitOr(int32_t a, int32_t b) { return (uint32_t)a | (uint32_t)b; }

int32_t BitXor(int32_t a, int32_t b) { return (uint32_t)a ^ (uint32_t)b; }

int32_t BitNot(int32_t a) { return ~(uint32_t)a; }

int BitsSelect(int mask, int bound, int val) { return BitXor(BitAnd(mask, bound), BitAnd(BitNot(mask), val)); }

int ConstantPOT(int fractional_bits, int exponent) { return (1 << (uint32_t)(fractional_bits + exponent)); }

int32_t MaskIfNonZero(int32_t a) { return a ? BitNot(0) : 0; }

int32_t MaskIfZero(int32_t a) { return MaskIfNonZero(!a); }

int32_t MaskIfLessThan(int32_t a, int32_t b) { return MaskIfNonZero((a < b)); }

int CountLeadingZeroBits(uint32_t x) {
#if defined(__GUNC__)
  return x ? __builtin_clz(x) : 8 * sizeof(uint32_t);
#else
  if (x == 0) {
    return 8 * sizeof(uint32_t);
  }
  const int32_t leading_positive = (uint32_t)(1) << (8 * sizeof(uint32_t) - 1);
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

int SaturatingRoundingMultiplyByPOT(int32_t x, int exponent) {
  if (exponent > 0) {
    const int min = INT32_MIN;
    const int max = INT32_MAX;
    const int scalar_int_bits = 8 * sizeof(int32_t);
    const int threshold = ((1 << (uint32_t)(scalar_int_bits - 1 - exponent)) - 1);
    const int positive_mask = x > threshold ? BitNot(0) : 0;
    const int negative_mask = x < -threshold ? BitNot(0) : 0;
    int result = x * ((int32_t)(1) << (uint32_t)exponent);
    result = BitsSelect(positive_mask, max, result);
    result = BitsSelect(negative_mask, min, result);
    return result;
  } else if (exponent < 0) {
    return RoundingDivideByPOT(x, -exponent);
  } else {
    return x;
  }
}

int32_t Rescale(int x, int integer_bits_src, int integer_bits_dst) {
  int exponent = integer_bits_src - integer_bits_dst;
  return SaturatingRoundingMultiplyByPOT(x, exponent);
}

int32_t reciprocal_on_interval_between_0_1(int32_t a) {
  int one = FixedPoint_One(0, FractionsBits(0));
  int half_sum = RoundingHalfSum(a, one);
  const int constant_48_over_17 = 1515870810;
  const int constant_neg_32_over_17 = -1010580540;
  int x = constant_48_over_17 + SaturatingRoundingDoublingHighMul(half_sum, constant_neg_32_over_17);
  for (int i = 0; i < 3; i++) {
    int half_sum_times_x = SaturatingRoundingDoublingHighMul(half_sum, x);
    int one_minus_half_sum_times_x = FixedPoint_One(2, FractionsBits(2)) - half_sum_times_x;
    x = x + Rescale(SaturatingRoundingDoublingHighMul(x, one_minus_half_sum_times_x), 2 + 2, 2);
  }
  return Rescale(x, 2 - 1, 0);
}

int32_t ComputerReciprocal(int32_t x, int x_digits, int *recip_shift) {
  int leading_zreos_plus_one = CountLeadingZeroBits((uint32_t)x);
  *recip_shift = x_digits - leading_zreos_plus_one;
  const int32_t shifted_minus_one = (int32_t)(((uint32_t)x << leading_zreos_plus_one) - ((uint32_t)(1) << 31));
  const int32_t shifted_scaled = reciprocal_on_interval_between_0_1(shifted_minus_one);
  return shifted_scaled;
}

int exp_on_interval_values(int a) {
  const int constant_neg_1_over_8 = 1895147668;
  const int constant_1_over_3 = 715827883;
  int fractional_bits = FractionsBits(0);
  int x = a + ConstantPOT(fractional_bits, -3);
  int x2 = SaturatingRoundingDoublingHighMul(x, x);
  int x3 = SaturatingRoundingDoublingHighMul(x2, x);
  int x4 = SaturatingRoundingDoublingHighMul(x2, x2);
  int x4_over_4 = SaturatingRoundingMultiplyByPOT(x4, -2);
  int x4_over_24_plus_x3_over_6_plus_x2_over_2 =
    SaturatingRoundingMultiplyByPOT((SaturatingRoundingDoublingHighMul((x4_over_4 + x3), constant_1_over_3) + x2), -1);
  return constant_neg_1_over_8 +
         SaturatingRoundingDoublingHighMul(constant_neg_1_over_8, (x + x4_over_24_plus_x3_over_6_plus_x2_over_2));
}

void exp_barrel_shifter(int exponent, int muliplier, int integer_bits, int fractional_bits, int remainder,
                        int *result) {
  if (integer_bits > exponent) {
    int total_shift = integer_bits > exponent ? fractional_bits + exponent : 0;
    *result = BitsSelect(MaskIfNonZero(BitAnd(remainder, (1 << (uint32_t)total_shift))),
                         SaturatingRoundingDoublingHighMul(*result, muliplier), *result);
  }
}

int exp_on_negative_values(int a, const int integer_bits) {
  int fractional_bits = FractionsBits(integer_bits);
  const int one_quarter = ConstantPOT(fractional_bits, -2);
  int a_mod_quarter_minus_one_quarter = ((unsigned)(a) & (one_quarter - 1)) - one_quarter;
  int result = exp_on_interval_values(Rescale(a_mod_quarter_minus_one_quarter, integer_bits, 0));
  int remainder = a_mod_quarter_minus_one_quarter - a;

  exp_barrel_shifter(-2, 1672461947, integer_bits, fractional_bits, remainder, &result);
  exp_barrel_shifter(-1, 1302514674, integer_bits, fractional_bits, remainder, &result);
  exp_barrel_shifter(+0, 790015084, integer_bits, fractional_bits, remainder, &result);
  exp_barrel_shifter(+1, 290630308, integer_bits, fractional_bits, remainder, &result);
  exp_barrel_shifter(+2, 39332535, integer_bits, fractional_bits, remainder, &result);
  exp_barrel_shifter(+3, 720401, integer_bits, fractional_bits, remainder, &result);
  exp_barrel_shifter(+4, 242, integer_bits, fractional_bits, remainder, &result);

  int clamp_bits = integer_bits > 5 ? 36 - integer_bits : 0;
  if (integer_bits > 5) {
    const int clamp = -(1 << (uint32_t)clamp_bits);
    result = BitsSelect(MaskIfLessThan(a, clamp), 0, result);
  }
  result = BitsSelect(MaskIfZero(a), FixedPoint_One(0, fractional_bits), result);
  return result;
}

void GetSqrtQuantMultiplierExp(int32_t input, int reverse_shift, int32_t *multiplier, int32_t *shift) {
  if (input <= 1) {
    *multiplier = INT_MAX;
    *shift = 0;
  }
  *shift = 11;
  while (input >= (1 << 29)) {
    input /= 4;
    ++*shift;
  }
  int max_left_shift_bits = CountLeadingSignBits(input);
  int left_shift_bit_pairs = max_left_shift_bits / 2 - 1;
  *shift -= left_shift_bit_pairs;
  input <<= 2 * left_shift_bit_pairs;
  int32_t fixedpoint_f3_input = input >> 1;  // sign: 1 bit, integer: 3 bit, fractional: 28 bit
  int32_t fp_f3_half_input = SaturatingRoundingMultiplyByPOT(fixedpoint_f3_input, -1);
  int32_t fp_f3_half_three = (1 << 28) + (1 << 27);
  int32_t tmp = (1 << 28);  // one
  for (int i = 0; i < 5; i++) {
    int32_t tmp3 = Rescale(SaturatingRoundingDoublingHighMul(tmp, SaturatingRoundingDoublingHighMul(tmp, tmp)), 9, 3);
    tmp = Rescale(SaturatingRoundingDoublingHighMul(fp_f3_half_three, tmp) -
                    SaturatingRoundingDoublingHighMul(fp_f3_half_input, tmp3),
                  6, 3);
  }
  const int32_t fp_f0_half_sqrt_2 = 1518500250;  // sqrt(2) / 2
  tmp = SaturatingRoundingDoublingHighMul(tmp, fp_f0_half_sqrt_2);
  *multiplier = tmp;
  if (*shift < 0) {
    *multiplier <<= -*shift;
    *shift = 0;
  }
  *shift *= reverse_shift;
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
