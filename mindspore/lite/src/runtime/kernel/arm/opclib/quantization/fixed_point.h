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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_FIXED_POINT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_FIXED_POINT_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

// Part 1: Low-level integer-arithmetic primitives.
// The implementations here are generic implementations valid for
// scalar types (e.g. std::int32_t). Architecture-specific SIMD types
// (e.g. NEON int32x4_t) may be supported by providing
// specializations for them in separate files.
//
// The purpose of these primitives is two-fold:
//  - They will be used to implement higher-level fixed-point
//    abstractions, namely the FixedPoint class and its arithmetic
//    operators.
//  - They will be directly used to implement some more involved
//    fixed-point computations, e.g. the fixed-point implementation
//    of math functions such as tanh.

// Some compile-time traits around raw types to handle SIMD aspects:
// number of lanes, underlying scalar type.
template <typename tIntegerType>
struct FixedPointRawTypeTraits {};

template <>
struct FixedPointRawTypeTraits<std::int32_t> {
  typedef std::int32_t ScalarRawType;
  static constexpr int kLanes = 1;
};

template <>
struct FixedPointRawTypeTraits<std::int16_t> {
  typedef std::int16_t ScalarRawType;
  static constexpr int kLanes = 1;
};

// Returns a SIMD value duplicating a scalar value across all lanes.
template <typename tRawType>
tRawType Dup(typename FixedPointRawTypeTraits<tRawType>::ScalarRawType x) {
  return x;
}

// Plain bit-wise AND
template <typename tIntegerType>
tIntegerType BitAnd(tIntegerType a, tIntegerType b) {
  return a & b;
}

// Plain bit-wise OR
template <typename tIntegerType>
tIntegerType BitOr(tIntegerType a, tIntegerType b) {
  return a | b;
}

// Plain bit-wise XOR
template <typename tIntegerType>
tIntegerType BitXor(tIntegerType a, tIntegerType b) {
  return a ^ b;
}

// Plain bit-wise NOT
template <typename tIntegerType>
tIntegerType BitNot(tIntegerType a) {
  return ~a;
}

// Integer addition. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Add(tIntegerType a, tIntegerType b) {
  return a + b;
}

// Integer multiplication. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Mul(tIntegerType a, tIntegerType b) {
  return a * b;
}

// Integer subtraction. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Sub(tIntegerType a, tIntegerType b) {
  return a - b;
}

// Integer unary negative. Not saturating. Overflow is undefined behavior.
template <typename tIntegerType>
tIntegerType Neg(tIntegerType a) {
  return -a;
}

// Integer arithmetic left-shift, equivalent to multiplying with a power of two.
// Negative values are OK. In case of overflow, no Undefined
// Behavior, but the results are implementation-defined (in practice,
// they currently are saturated, but we make no commitment to that). The idea
// is that the caller will want to implement the overflowing cases with
// saturation with compare-and-mask, so we don't care about the results
// in the overflow case, we just want to avoid undefined behavior.
//
// tIntegerType may be int32 or any narrower signed type.
template <typename tIntegerType, typename OffsetType>
tIntegerType ShiftLeft(tIntegerType a, OffsetType offset) {
  const std::int64_t wide_a = (std::int64_t)(a);
  const std::int64_t wide_shifted = wide_a * (1 << offset);
  const auto min = std::numeric_limits<tIntegerType>::min();
  const auto max = std::numeric_limits<tIntegerType>::max();
  return wide_shifted < min ? min : wide_shifted > max ? max : (tIntegerType)(wide_shifted);
}

// Integer arithmetic right-shift. Not rounding.
// Relying on implementation-defined, but in-practice-consistent,
// C++ compiler behavior.
template <typename tIntegerType>
tIntegerType ShiftRight(tIntegerType a, int offset) {
  return a >> offset;
}

// Each bit of the result is set to the corresponding bit of either then_val or
// else_val depending on whether the corresponding bit of if_mask is set.
// Equivalent to the VBSL instruction in ARM NEON.
template <typename tIntegerType>
tIntegerType SelectUsingMask(tIntegerType if_mask, tIntegerType then_val, tIntegerType else_val) {
  return BitXor(BitAnd(if_mask, then_val), BitAnd(BitNot(if_mask), else_val));
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is non-zero.
template <typename tIntegerType>
tIntegerType MaskIfNonZero(tIntegerType a) {
  static constexpr tIntegerType zero = 0;
  return a ? BitNot(zero) : zero;
}

// For each input scalar, the corresponding bits of the result are set if the
// input scalar is zero.
template <typename tIntegerType>
tIntegerType MaskIfZero(tIntegerType a) {
  return MaskIfNonZero<tIntegerType>(!a);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are equal.
template <typename tIntegerType>
tIntegerType MaskIfEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a == b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars are not equal.
template <typename tIntegerType>
tIntegerType MaskIfNotEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a != b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a > b.
template <typename tIntegerType>
tIntegerType MaskIfGreaterThan(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a > b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a >= b.
template <typename tIntegerType>
tIntegerType MaskIfGreaterThanOrEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a >= b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a < b.
template <typename tIntegerType>
tIntegerType MaskIfLessThan(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a < b);
}

// For each pair of input scalars, the corresponding bits of the result are
// set if the input scalars a, b satisfy a <= b.
template <typename tIntegerType>
tIntegerType MaskIfLessThanOrEqual(tIntegerType a, tIntegerType b) {
  return MaskIfNonZero<tIntegerType>(a <= b);
}

// Returns true if all of the input scalars are nonzero.
// This function may currently assume that each of the input scalars has either
// all or none of its bits set. Otherwise, its behavior is currently undefined.
template <typename tIntegerType>
bool All(tIntegerType a) {
  return a;
}

// Returns true if any of the input scalars are nonzero.
// This function may currently assume that each of the input scalars has either
// all or none of its bits set. Otherwise, its behavior is currently undefined.
template <typename tIntegerType>
bool Any(tIntegerType a) {
  return a;
}

// Returns (a+b)/2, rounded to the nearest integer.
// Equivalent to VRHADD in the ARM NEON instruction set.
template <typename IntegerType>
IntegerType RoundingHalfSum(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  (void)b;
  return a;
}

template <>
inline std::int32_t RoundingHalfSum(std::int32_t a, std::int32_t b) {
  std::int64_t a64 = a;
  std::int64_t b64 = b;
  std::int64_t sum = a64 + b64;
  std::int64_t sign = sum >= 0 ? 1 : -1;
  return (std::int32_t)((sum + sign) / 2);
}

template <>
inline std::int16_t RoundingHalfSum(std::int16_t a, std::int16_t b) {
  std::int32_t a32 = a;
  std::int32_t b32 = b;
  std::int32_t sum = a32 + b32;
  std::int32_t sign = sum >= 0 ? 1 : -1;
  return (std::int16_t)((sum + sign) / 2);
}

template <typename IntegerType>
IntegerType SaturatingAdd(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  (void)b;
  return a;
}

// So far this is only needed for int16.
template <>
inline std::int16_t SaturatingAdd(std::int16_t a, std::int16_t b) {
  std::int32_t a32 = a;
  std::int32_t b32 = b;
  std::int32_t sum = a32 + b32;
  return (std::int16_t)(std::min((std::int32_t)(32767), std::max((std::int32_t)(-32768), sum)));
}

template <>
inline std::int8_t SaturatingAdd(std::int8_t a, std::int8_t b) {
  std::int16_t a16 = a;
  std::int16_t b16 = b;
  std::int16_t sum = a16 + b16;
  return (std::int8_t)(std::min((int16_t)(std::numeric_limits<int8_t>::max()),
                                std::max((int16_t)(std::numeric_limits<int8_t>::min()), sum)));
}

// Returns a+b, saturating if the integers are 16bit or narrower,
// otherwise just a plain addition.
template <typename IntegerType, bool Is16Bit>
struct AddSaturatingIf16BitImpl {
  static IntegerType Run(IntegerType a, IntegerType b) { return Add(a, b); }
};
template <typename IntegerType>
struct AddSaturatingIf16BitImpl<IntegerType, true> {
  static IntegerType Run(IntegerType a, IntegerType b) { return SaturatingAdd(a, b); }
};
template <typename IntegerType>
IntegerType AddSaturatingIf16Bit(IntegerType a, IntegerType b) {
  using ScalarType = typename FixedPointRawTypeTraits<IntegerType>::ScalarRawType;
  return AddSaturatingIf16BitImpl<IntegerType, sizeof(ScalarType) == 2>::Run(a, b);
}

// Returns the integer that represents the product of two fixed-point
// numbers, interpreting all integers as fixed-point values in the
// interval [-1, 1), rounding to the nearest value, and saturating
// -1 * -1 to the maximum value (since 1 is not in the half-open
// interval [-1, 1)).
//
// [The explanation below specializes to std::int32_t for example purpose.]
//
// The mapping between IntegerType and the interval [-1, 1) is unique and
// implied by IntegerType, which is assumed to be signed. For example,
// for IntegerType==std::int32_t, the mapping is
//   real_value = integer_value / 2^31.
// So in this case, and leaving aside rounding and saturating, this
// function computes ((a / 2^31) * (b / 2^31)) * 2^31, which simplifies to
//   (a * b) / 2^31.
//
// The 'doubling' part in the name of this function comes from the fact that
// this operation is very close to a "multiply-high" operation, keeping only
// the top half bits, except that that would be effectively computing
//   (a * b) / 2^32,
// so here we are computing 2x that, since
//   1/2^31 = 2 * 1/2^32.
// The idea is to use all of the available 32 bits in the destination int32
// value.
//
// [End of the explanation specializing to int32.]
//
// This is equivalent to the VQRDMULH instruction in ARM NEON.
template <typename IntegerType>
IntegerType SaturatingRoundingDoublingHighMul(IntegerType a, IntegerType b) {
  static_assert(std::is_same<IntegerType, void>::value, "unimplemented");
  (void)b;
  return a;
}

// This function implements the same computation as the ARMv7 NEON VQRDMULH
// instruction.
template <>
inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 = (std::int32_t)((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

template <>
inline std::int16_t SaturatingRoundingDoublingHighMul(std::int16_t a, std::int16_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int16_t>::min();
  std::int32_t a_32(a);
  std::int32_t b_32(b);
  std::int32_t ab_32 = a_32 * b_32;
  std::int16_t nudge = ab_32 >= 0 ? (1 << 14) : (1 - (1 << 14));
  std::int16_t ab_x2_high16 = (std::int16_t)((ab_32 + nudge) / (1 << 15));
  return overflow ? std::numeric_limits<std::int16_t>::max() : ab_x2_high16;
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
template <typename IntegerType, typename ExponentType>
inline IntegerType RoundingDivideByPOT(IntegerType x, ExponentType exponent) {
  assert(exponent >= 0);
  assert(exponent <= 31);
  const IntegerType mask = Dup<IntegerType>((1ll << exponent) - 1);
  const IntegerType zero = Dup<IntegerType>(0);
  const IntegerType one = Dup<IntegerType>(1);
  const IntegerType remainder = BitAnd(x, mask);
  const IntegerType threshold = Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
  return Add(ShiftRight(x, exponent), BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}

inline int MultiplyByQuantizedMultiplier(int32_t value, int32_t multiplier, int32_t left_shift, int32_t right_shift) {
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(value * (1 << left_shift), multiplier), -right_shift);
}

// Returns the product of a run-time integer value by a compile-time power
// of two, with either a positive exponent (equivalent to an arithmetic
// left shift, saturating) or a negative exponent (equivalent to an arithmetic
// right shift, rounding to nearest).
template <int Exponent, typename IntegerType, int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0)>
struct ImplSaturatingRoundingMultiplyByPOT {};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 0> {
  static IntegerType eval(IntegerType x) { return x; }
};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, 1> {
  static IntegerType eval(IntegerType x) {
    using ScalarIntegerType = typename FixedPointRawTypeTraits<IntegerType>::ScalarRawType;
    const IntegerType min = Dup<IntegerType>(std::numeric_limits<ScalarIntegerType>::min());
    const IntegerType max = Dup<IntegerType>(std::numeric_limits<ScalarIntegerType>::max());
    const int ScalarIntegerTypeBits = 8 * sizeof(ScalarIntegerType);

    const std::int32_t threshold = ((1 << (ScalarIntegerTypeBits - 1 - Exponent)) - 1);
    const IntegerType positive_mask = MaskIfGreaterThan(x, Dup<IntegerType>(threshold));
    const IntegerType negative_mask = MaskIfLessThan(x, Dup<IntegerType>(-threshold));

    IntegerType result = ShiftLeft(x, Exponent);
    result = SelectUsingMask(positive_mask, max, result);
    result = SelectUsingMask(negative_mask, min, result);
    return result;
  }
};

template <int Exponent, typename IntegerType>
struct ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType, -1> {
  static IntegerType eval(IntegerType x) { return RoundingDivideByPOT<IntegerType>(x, -Exponent); }
};

template <int Exponent, typename IntegerType>
IntegerType SaturatingRoundingMultiplyByPOT(IntegerType x) {
  return ImplSaturatingRoundingMultiplyByPOT<Exponent, IntegerType>::eval(x);
}

// Part 2: the FixedPoint class.

// A FixedPoint object represents a fixed-point value stored in the underlying
// integer type tRawType, if tRawType is a plain scalar integer type.
// Alternatively, tRawType may be a SIMD type (e.g. NEON int32x4_t) in which
// case a FixedPoint object represents a corresponding SIMD vector of fixed
// point values.
//
// tIntegerBits describes the range of the fixed-point format: if
// tIntegerBits == m then the range of representable values is the half-open
// interval [-2^m; 2^m) where the open boundary on the right side means that
// 2^m is not representable (how close the maximum representable value is to
// it, depends on bit-depth of tRawType).
//
// In "Q format notation",
//   https://en.wikipedia.org/wiki/Q_(number_format)
// we are describing the format
//   Qm.n
// where
//   m = tIntegerBits
// and
//   n = NumberOfBits(tRawType) - (m + 1)
// Note that the (m + 1) in the above line is because we adopt the convention
// that we count the integer bits exclusively of the sign bit; so (m + 1) is
// the total number of integer bits inclusive of the sign bit.
//
// Accordingly, the number of integral representable values in our range
//   [-2^m ; 2^m)
// is equal to 2^(m+1).
template <typename tRawType, int tIntegerBits>
class FixedPoint {
 public:
  typedef tRawType RawType;

  typedef FixedPointRawTypeTraits<RawType> RawTypeTraits;
  typedef typename RawTypeTraits::ScalarRawType ScalarRawType;

  static constexpr int kTotalBits = 8 * sizeof(ScalarRawType);
  static constexpr int kIntegerBits = tIntegerBits;
  static constexpr int kFractionalBits = kTotalBits - 1 - kIntegerBits;
  static_assert(kIntegerBits >= 0 && kIntegerBits < kTotalBits, "bad IntegerBits");

  typedef FixedPoint<ScalarRawType, kIntegerBits> ScalarFixedPointType;

  static const ScalarRawType ScalarRawMin() { return std::numeric_limits<ScalarRawType>::min(); }

  static const ScalarRawType ScalarRawMax() { return std::numeric_limits<ScalarRawType>::max(); }

  static const ScalarRawType RawMin() { return VectorFromScalar(ScalarRawMin()); }

  static const ScalarRawType RawMax() { return VectorFromScalar(ScalarRawMax()); }

  static FixedPoint FromRaw(RawType x) {
    FixedPoint retval;
    retval.raw() = x;
    return retval;
  }

  static FixedPoint FromScalarRaw(ScalarRawType x) {
    FixedPoint retval;
    retval.raw() = Dup<RawType>(x);
    return retval;
  }

  static FixedPoint FromScalarFixedPoint(ScalarFixedPointType x) { return FromScalarRaw(x.raw()); }

  template <int Exponent>
  static FixedPoint ConstantPOT() {
    static constexpr int kOffset = kFractionalBits + Exponent;
    static_assert(kOffset < 31, "Constant not exactly representable in this fixed-point format");
    return FromScalarRaw(ScalarRawType(1) << kOffset);
  }

  static FixedPoint Zero() { return FromScalarRaw(0); }

  static FixedPoint One() {
    return FromScalarRaw(kIntegerBits == 0 ? ScalarRawMax()
                                           : (ScalarRawType(1) << (kIntegerBits == 0 ? 0 : kFractionalBits)));
  }

  static FixedPoint FromDouble(double x) {
    const double min_bound = (double)(ScalarRawMin());
    const double max_bound = (double)(ScalarRawMax());
    return FromScalarRaw(
      (ScalarRawType)(std::min(std::max(round(x * (double)(1ll << kFractionalBits)), min_bound), max_bound)));
  }

  RawType raw() const { return i_; }
  RawType &raw() { return i_; }

 private:
  RawType i_;
};

// Part 3: implementation of arithmetic operators for the
// FixedPoint class, and a few related functions.

// A FixedPoint multiplication is just a
// SaturatingRoundingDoublingHighMul operation on the underlying
// raw integer values. The IntegerBits simply add up, as is obvious
// from the fact that the range is [-2^IntegerBits, 2^IntegerBits).
template <typename tRawType, int tIntegerBits_a, int tIntegerBits_b>
FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> operator*(FixedPoint<tRawType, tIntegerBits_a> a,
                                                                FixedPoint<tRawType, tIntegerBits_b> b) {
  FixedPoint<tRawType, tIntegerBits_a + tIntegerBits_b> c;
  c.raw() = SaturatingRoundingDoublingHighMul(a.raw(), b.raw());
  return c;
}

// Tweaking IntegerBits gives exact multiplication by a power of two.
template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tExponent + tIntegerBits> ExactMulByPot(FixedPoint<tRawType, tIntegerBits> a) {
  FixedPoint<tRawType, tExponent + tIntegerBits> c;
  c.raw() = a.raw();
  return c;
}

// If we want to leave IntegerBits fixed, then multiplication
// by a power of two has to be saturating/rounding, not exact anymore.
template <int tExponent, typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SaturatingRoundingMultiplyByPOT(FixedPoint<tRawType, tIntegerBits> a) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(SaturatingRoundingMultiplyByPOT<tExponent>(a.raw()));
}

// Generic arithmetic operators.

#define MAKE_FIXEDPOINT_UNARY_FUNC(FuncName, ImplFuncName)                            \
  template <typename tRawType, int tIntegerBits>                                      \
  FixedPoint<tRawType, tIntegerBits> FuncName(FixedPoint<tRawType, tIntegerBits> a) { \
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(ImplFuncName(a.raw()));        \
  }

#define MAKE_FIXEDPOINT_BINARY_FUNC(FuncName, ImplFuncName)                             \
  template <typename tRawType, int tIntegerBits>                                        \
  FixedPoint<tRawType, tIntegerBits> FuncName(FixedPoint<tRawType, tIntegerBits> a,     \
                                              FixedPoint<tRawType, tIntegerBits> b) {   \
    return FixedPoint<tRawType, tIntegerBits>::FromRaw(ImplFuncName(a.raw(), b.raw())); \
  }

MAKE_FIXEDPOINT_UNARY_FUNC(operator-, Neg)
MAKE_FIXEDPOINT_UNARY_FUNC(operator~, BitNot)
MAKE_FIXEDPOINT_BINARY_FUNC(operator+, Add)
MAKE_FIXEDPOINT_BINARY_FUNC(operator-, Sub)
MAKE_FIXEDPOINT_BINARY_FUNC(operator&, BitAnd)
MAKE_FIXEDPOINT_BINARY_FUNC(operator^, BitXor)
MAKE_FIXEDPOINT_BINARY_FUNC(operator|, BitOr)
MAKE_FIXEDPOINT_BINARY_FUNC(RoundingHalfSum, RoundingHalfSum)

#undef MAKE_FIXEDPOINT_UNARY_FUNC
#undef MAKE_FIXEDPOINT_BINARY_FUNC

#define MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(FuncName)  \
  template <typename tRawType, int tIntegerBits>            \
  tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a) { \
    return FuncName(a.raw());                               \
  }

#define MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(FuncName)                                       \
  template <typename tRawType, int tIntegerBits>                                                  \
  tRawType FuncName(FixedPoint<tRawType, tIntegerBits> a, FixedPoint<tRawType, tIntegerBits> b) { \
    return FuncName(a.raw(), b.raw());                                                            \
  }

MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfZero)
MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW(MaskIfNonZero)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfNotEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfGreaterThanOrEqual)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThan)
MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW(MaskIfLessThanOrEqual)

#undef MAKE_FIXEDPOINT_UNARY_FUNC_RETURNING_RAW
#undef MAKE_FIXEDPOINT_BINARY_FUNC_RETURNING_RAW

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SelectUsingMask(tRawType if_mask, FixedPoint<tRawType, tIntegerBits> then_val,
                                                   FixedPoint<tRawType, tIntegerBits> else_val) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(SelectUsingMask(if_mask, then_val.raw(), else_val.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator==(FixedPoint<tRawType, tIntegerBits> a, FixedPoint<tRawType, tIntegerBits> b) {
  return All(MaskIfEqual(a.raw(), b.raw()));
}

template <typename tRawType, int tIntegerBits>
bool operator!=(FixedPoint<tRawType, tIntegerBits> a, FixedPoint<tRawType, tIntegerBits> b) {
  return !(a == b);
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> SaturatingAdd(FixedPoint<tRawType, tIntegerBits> a,
                                                 FixedPoint<tRawType, tIntegerBits> b) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(SaturatingAdd(a.raw(), b.raw()));
}

template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, tIntegerBits> AddSaturatingIf16Bit(FixedPoint<tRawType, tIntegerBits> a,
                                                        FixedPoint<tRawType, tIntegerBits> b) {
  return FixedPoint<tRawType, tIntegerBits>::FromRaw(AddSaturatingIf16Bit(a.raw(), b.raw()));
}

// Conversion to floating-point.
template <typename tRawType, int tIntegerBits>
double ToDouble(FixedPoint<tRawType, tIntegerBits> x) {
  static_assert(FixedPointRawTypeTraits<tRawType>::kLanes == 1, "not applicable to SIMD types");
  typedef FixedPoint<tRawType, tIntegerBits> F;
  return x.raw() / (double)(1ll << F::kFractionalBits);
}

// Rescale changes the number of IntegerBits and updates the underlying
// raw integer value accordingly.
template <int tIntegerBitsDst, typename tRawType, int tIntegerBitsSrc>
FixedPoint<tRawType, tIntegerBitsDst> Rescale(FixedPoint<tRawType, tIntegerBitsSrc> x) {
  static constexpr int kExponent = tIntegerBitsSrc - tIntegerBitsDst;
  FixedPoint<tRawType, tIntegerBitsDst> result;
  result.raw() = SaturatingRoundingMultiplyByPOT<kExponent>(x.raw());
  return result;
}

// CheckedFixedPointConstant allows to specify fixed-point constants
// initialized as real numbers, in a way that does not compile floating-point
// arithmetic in production code, yet still checks agreement with the
// floating-point expressions when asserts are enabled.
//
// The raw integer value provided is always a int32, encoding a 32-bit
// fixed-point value, regardless of the actual Scalar type. This allows
// writing generic code that applies just as well to the 32-bit and 16-bit
// cases. In the 16-bit case, the raw integer value is internally
// rounding-shifted by 16 bits to the right.
template <typename FixedPointType>
inline typename FixedPointType::ScalarRawType RescaleConstantInitializer(std::int32_t int32_value) {
  typedef typename FixedPointType::ScalarRawType ScalarRawType;
  static constexpr int ScalarTypeBits = 8 * sizeof(ScalarRawType);
  return (ScalarRawType)(RoundingDivideByPOT<std::int32_t>(int32_value, 32 - ScalarTypeBits));
}

// Implementation of exponential function.

// Returns -tanh(x) for x < 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> neg_tanh_on_negative_values(FixedPoint<tRawType, tIntegerBits> a) {
  return one_minus_x_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(ExactMulByPot<1>(a)));
}

// Returns tanh(x) for any x.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> tanh(FixedPoint<tRawType, tIntegerBits> a) {
  typedef FixedPoint<tRawType, tIntegerBits> InputF;
  typedef FixedPoint<tRawType, 0> ResultF;
  tRawType mask_if_negative = MaskIfLessThan(a, InputF::Zero());
  tRawType mask_if_zero = MaskIfZero(a);
  InputF n = SelectUsingMask(mask_if_negative, a, -a);
  ResultF t = neg_tanh_on_negative_values(n);
  return SelectUsingMask(mask_if_zero, ResultF::Zero(), SelectUsingMask(mask_if_negative, -t, t));
}

// Implementation of logistic function.

// Returns logistic(x) = 1 / (1 + exp(-x)) for x > 0.
template <typename tRawType, int tIntegerBits>
FixedPoint<tRawType, 0> logistic_on_positive_values(FixedPoint<tRawType, tIntegerBits> a) {
  return one_over_one_plus_x_for_x_in_0_1(exp_on_negative_values(-a));
}

#ifdef ENABLE_NEON
inline int32x4_t RoundingDivideByPOTInt32x4(int32x4_t x, int exponent) {
  const int32x4_t shift_vec = vdupq_n_s32(-exponent);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
  const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
  return vrshlq_s32(fixed_up_x, shift_vec);
}

inline int32x4_t SaturatingRoundingDoublingHighMulInt32x4(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_QUANTIZATION_FIXED_POINT_H_

