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
#ifndef MINDSPORE_CORE_BASE_FLOAT16_H_
#define MINDSPORE_CORE_BASE_FLOAT16_H_

#if defined(ENABLE_ARM32) || defined(ENABLE_ARM64)
// Built for lite and ARM
#include <arm_neon.h>

using float16 = float16_t;

#else
#include <cmath>
#include <climits>
#include <cstdint>
#include <ostream>
#include <limits>
#include <functional>

// Implement Float16 for mindspore, inspired by Eigen::half.
namespace mindspore {
class Float16 {
 public:
  static constexpr uint16_t value_mask = 0x7fff;
  static constexpr uint16_t nan_value = 0x7e00;
  static constexpr uint16_t inf_value = 0x7c00;
  static constexpr uint16_t true_value = 0x3c00;

  union Union32 {
    uint32_t u;
    float f;
  };

  Float16() = default;
  ~Float16() = default;

  Float16(const Float16 &other) noexcept = default;
  Float16(Float16 &&other) noexcept = default;

  Float16 &operator=(const Float16 &other) noexcept = default;
  Float16 &operator=(Float16 &&other) noexcept = default;

  static Float16 FromRaw(uint16_t v) {
    Float16 f;
    f.value_ = v;
    return f;
  }

  explicit Float16(float f) : value_(FromFloat32(f)) {}
  explicit Float16(bool b) : value_(b ? true_value : 0) {}
  template <typename T>
  explicit Float16(const T &v) : value_(FromFloat32(static_cast<float>(v))) {}

  uint16_t int_value() const { return value_; }

  explicit operator bool() const { return (value_ & value_mask) != 0; }
  explicit operator float() const { return ToFloat32(*this); }
  explicit operator double() const { return static_cast<double>(ToFloat32(*this)); }
  explicit operator int8_t() const { return static_cast<int8_t>(ToFloat32(*this)); }
  explicit operator uint8_t() const { return static_cast<uint8_t>(ToFloat32(*this)); }
  explicit operator int16_t() const { return static_cast<int16_t>(ToFloat32(*this)); }
  explicit operator uint16_t() const { return static_cast<uint16_t>(ToFloat32(*this)); }
  explicit operator int32_t() const { return static_cast<int32_t>(ToFloat32(*this)); }
  explicit operator uint32_t() const { return static_cast<uint32_t>(ToFloat32(*this)); }
  explicit operator int64_t() const { return static_cast<int64_t>(ToFloat32(*this)); }
  explicit operator uint64_t() const { return static_cast<uint64_t>(ToFloat32(*this)); }

  Float16 &operator+=(const Float16 &b) {
    value_ = FromFloat32(ToFloat32(*this) + ToFloat32(b));
    return *this;
  }

  Float16 &operator-=(const Float16 &b) {
    value_ = FromFloat32(ToFloat32(*this) - ToFloat32(b));
    return *this;
  }

  Float16 &operator*=(const Float16 &b) {
    value_ = FromFloat32(ToFloat32(*this) * ToFloat32(b));
    return *this;
  }

  Float16 &operator/=(const Float16 &b) {
    value_ = FromFloat32(ToFloat32(*this) / ToFloat32(b));
    return *this;
  }

  static float ToFloat32(Float16 f16) {
    constexpr Union32 magic = {113 << 23};
    constexpr uint32_t exponent_adjust = ((127 - 15) << 23);
    constexpr uint32_t inf_extra_exp_adjust = ((128 - 16) << 23);
    constexpr uint32_t zero_extra_exp_adjust = (1 << 23);
    constexpr uint32_t sign_mask = 0x8000;
    constexpr unsigned int shifted_exp = (0x7c00 << 13);  // Exponent mask after shift.
    constexpr unsigned int exponent_bits = 13;
    constexpr unsigned int sign_bit_shift = 16;
    // Exponent/mantissa bits.
    Union32 f32;
    f32.u = (static_cast<uint32_t>(f16.value_ & value_mask) << exponent_bits);
    // Just the exponent.
    unsigned int exp = (shifted_exp & f32.u);
    f32.u += exponent_adjust;
    // Handle exponent special cases.
    if (exp == shifted_exp) {
      // Inf/NaN, extra exp adjust.
      f32.u += inf_extra_exp_adjust;
    } else if (exp == 0) {
      // Zero/Denormal, extra exp adjust and renormalize.
      f32.u += zero_extra_exp_adjust;
      f32.f -= magic.f;
    }
    // Set sign bit.
    f32.u |= ((f16.value_ & sign_mask) << sign_bit_shift);
    return f32.f;
  }

 private:
  static uint16_t FromFloat32(float f32) {
    constexpr uint32_t magic = {113 << 23};
    constexpr Union32 f32infty = {255 << 23};
    constexpr Union32 f16max = {(127 + 16) << 23};
    constexpr Union32 denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
    constexpr unsigned int exponent_bits = 13;
    constexpr unsigned int sign_bit_shift = 16;
    constexpr unsigned int sign_mask = 0x80000000u;
    constexpr uint32_t rouding_bias_part1 = ((unsigned int)(15 - 127) << 23) + 0xfff;

    Union32 f;
    f.f = f32;
    unsigned int sign = f.u & sign_mask;
    f.u ^= sign;
    uint16_t result = 0;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).
    if (f.u >= f16max.u) {
      // Result is Inf or NaN (all exponent bits set).
      result = (f.u > f32infty.u) ? nan_value : inf_value;
    } else if (f.u < magic) {
      // (De)normalized number or zero; resulting FP16 is subnormal or zero.
      // Use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;
      // And one integer subtract of the bias later, we have our final float!
      result = static_cast<uint16_t>(f.u - denorm_magic.u);
    } else {
      // Resulting mantissa is odd.
      unsigned int mant_odd = (f.u >> exponent_bits) & 1;
      // Update exponent, rounding bias part 1;
      f.u += rouding_bias_part1;
      // Rounding bias part 2;
      f.u += mant_odd;
      // Take the bits!
      result = static_cast<uint16_t>(f.u >> exponent_bits);
    }
    // Set sign bit.
    result |= static_cast<uint16_t>(sign >> sign_bit_shift);
    return result;
  }

  uint16_t value_;
};

inline Float16 operator+(const Float16 &a, const Float16 &b) {
  return Float16(static_cast<float>(a) + static_cast<float>(b));
}

inline Float16 operator*(const Float16 &a, const Float16 &b) {
  return Float16(static_cast<float>(a) * static_cast<float>(b));
}

inline Float16 operator-(const Float16 &a, const Float16 &b) {
  return Float16(static_cast<float>(a) - static_cast<float>(b));
}

inline Float16 operator/(const Float16 &a, const Float16 &b) {
  return Float16(static_cast<float>(a) / static_cast<float>(b));
}

// Division by an size_t. Do it in full float precision to avoid
// accuracy issues in converting the denominator to float16.
inline Float16 operator/(const Float16 &a, size_t b) { return Float16(static_cast<float>(a) / static_cast<float>(b)); }

inline Float16 operator-(const Float16 &a) {
  constexpr uint16_t sign_mask = 0x8000;
  return Float16::FromRaw(a.int_value() ^ sign_mask);
}

inline bool operator==(const Float16 &a, const Float16 &b) {
  return std::equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator!=(const Float16 &a, const Float16 &b) {
  return std::not_equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator<(const Float16 &a, const Float16 &b) { return static_cast<float>(a) < static_cast<float>(b); }
inline bool operator<=(const Float16 &a, const Float16 &b) { return static_cast<float>(a) <= static_cast<float>(b); }
inline bool operator>(const Float16 &a, const Float16 &b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator>=(const Float16 &a, const Float16 &b) { return static_cast<float>(a) >= static_cast<float>(b); }

inline std::ostream &operator<<(std::ostream &os, const Float16 &v) { return (os << static_cast<float>(v)); }

}  // namespace mindspore

using float16 = mindspore::Float16;

namespace std {
template <>
struct hash<float16> {
  std::size_t operator()(const float16 &f16) const noexcept { return static_cast<std::size_t>(f16.int_value()); }
};

template <>
struct numeric_limits<float16> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr std::float_denorm_style has_denorm = std::denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr std::float_round_style round_style = std::round_to_nearest;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = false;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr bool traps = true;
  static constexpr bool tinyness_before = false;

  static constexpr uint16_t raw_min = 0x400;
  static constexpr uint16_t raw_max = 0x7bff;
  static constexpr uint16_t raw_lowest = 0xfbff;
  static constexpr uint16_t raw_epsilon = 0x0800;
  static constexpr float round_error_value = 0.5;

  static float16(min)() noexcept { return float16::FromRaw(raw_min); }
  static float16(max)() noexcept { return float16::FromRaw(raw_max); }
  static float16 lowest() noexcept { return float16::FromRaw(raw_lowest); }
  static float16 epsilon() noexcept { return float16::FromRaw(raw_epsilon); }
  static float16 round_error() noexcept { return float16(round_error_value); }
  static float16 infinity() noexcept { return float16::FromRaw(float16::inf_value); }
  static float16 quiet_NaN() noexcept { return float16::FromRaw(float16::nan_value); }
  static float16 signaling_NaN() noexcept { return float16::FromRaw(float16::nan_value); }
  static float16 denorm_min() noexcept { return float16::FromRaw(1); }
};

// If std::numeric_limits<T> is specialized, should also specialize
// std::numeric_limits<const T>, std::numeric_limits<volatile T>, and
// std::numeric_limits<const volatile T>
// https://stackoverflow.com/a/16519653/
template <>
struct numeric_limits<const mindspore::Float16> : numeric_limits<mindspore::Float16> {};
template <>
struct numeric_limits<volatile mindspore::Float16> : numeric_limits<mindspore::Float16> {};
template <>
struct numeric_limits<const volatile mindspore::Float16> : numeric_limits<mindspore::Float16> {};
}  // namespace std

// Implements standard math functions for float16.
inline bool(isinf)(const float16 &a) { return (a.int_value() & float16::value_mask) == float16::inf_value; }
inline bool(isnan)(const float16 &a) { return (a.int_value() & float16::value_mask) > float16::inf_value; }
inline bool(isfinite)(const float16 &a) { return !(isinf(a)) && !(isnan(a)); }
inline float16 abs(const float16 &a) { return float16::FromRaw(a.int_value() & float16::value_mask); }
inline float16 exp(const float16 &a) { return float16(::expf(static_cast<float>(a))); }
inline float16 log(const float16 &a) { return float16(::logf(static_cast<float>(a))); }
inline float16 log1p(const float16 &a) { return float16(::log1pf(static_cast<float>(a))); }
inline float16 log10(const float16 &a) { return float16(::log10f(static_cast<float>(a))); }
inline float16 sqrt(const float16 &a) { return float16(::sqrtf(static_cast<float>(a))); }
inline float16 sin(const float16 &a) { return float16(::sinf(static_cast<float>(a))); }
inline float16 cos(const float16 &a) { return float16(::cosf(static_cast<float>(a))); }
inline float16 tan(const float16 &a) { return float16(::tanf(static_cast<float>(a))); }
inline float16 tanh(const float16 &a) { return float16(::tanhf(static_cast<float>(a))); }
inline float16 floor(const float16 &a) { return float16(::floorf(static_cast<float>(a))); }
inline float16 ceil(const float16 &a) { return float16(::ceilf(static_cast<float>(a))); }
inline float16(min)(const float16 &a, const float16 &b) { return b < a ? b : a; }
inline float16(max)(const float16 &a, const float16 &b) { return a < b ? b : a; }
inline float16 pow(const float16 &a, const float16 &b) {
  return float16(::powf(static_cast<float>(a), static_cast<float>(b)));
}

#endif  // ENABLE_ARM32 || ENABLE_ARM64

inline float half_to_float(float16 h) { return static_cast<float>(h); }

#endif  // MINDSPORE_CORE_BASE_FLOAT16_H_
