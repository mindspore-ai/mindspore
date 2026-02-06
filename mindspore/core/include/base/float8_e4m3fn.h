/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_BASE_FLOAT8_E4M3FN_H_
#define MINDSPORE_CORE_BASE_FLOAT8_E4M3FN_H_

#include <type_traits>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <limits>
#include <functional>
#include "mindapi/base/macros.h"

// Implement float8_e4m3fn for mindspore
namespace mindspore {
class MS_CORE_API Float8_e4m3fn {
 public:
  static constexpr uint8_t value_mask = 0x7F;
  static constexpr uint8_t exponent_mask = 0x7C;
  static constexpr uint8_t mantissa_mask = 0x03;
  static constexpr uint8_t sign_mask = 0x80;
  static constexpr uint8_t inf_value = 0x7C;
  static constexpr uint8_t nan_value = 0x7F;
  static constexpr uint8_t true_value = 0x04;
  static constexpr uint32_t f32_inf_value = 0x7F800000;
  static constexpr uint32_t f32_mantissa_mask = 0x007FFFFF;
  static constexpr uint32_t f32_sign_mask = 0x80000000;
  static constexpr uint32_t f32_nan_value = 0x7fc00000;
  static constexpr uint32_t f32_zero_value = 0x00000000;

  union Union32 {
    uint32_t u;
    float f;
  };

  Float8_e4m3fn() = default;
  ~Float8_e4m3fn() = default;

  Float8_e4m3fn(const Float8_e4m3fn &other) noexcept = default;
  Float8_e4m3fn(Float8_e4m3fn &&other) noexcept = default;

  Float8_e4m3fn &operator=(const Float8_e4m3fn &other) noexcept = default;
  Float8_e4m3fn &operator=(Float8_e4m3fn &&other) noexcept = default;

  static Float8_e4m3fn FromRaw(uint8_t v) {
    Float8_e4m3fn f;
    f.value_ = v;
    return f;
  }

  explicit Float8_e4m3fn(float f) : value_(FromFloat32(f)) {}
  explicit Float8_e4m3fn(bool b) : value_(b ? true_value : 0) {}
  template <typename T>
  explicit Float8_e4m3fn(const T &v) : value_(FromFloat32(static_cast<float>(v))) {}

  uint8_t int_value() const { return value_; }

  template <typename T>
  explicit operator T() const {
    return static_cast<T>(ToFloat32(*this));
  }

  explicit operator bool() const { return (value_ & value_mask) != 0; }
  explicit operator float() const { return ToFloat32(*this); }

  Float8_e4m3fn &operator+=(const Float8_e4m3fn &b) {
    value_ = FromFloat32(ToFloat32(*this) + ToFloat32(b));
    return *this;
  }

  Float8_e4m3fn &operator-=(const Float8_e4m3fn &b) {
    value_ = FromFloat32(ToFloat32(*this) - ToFloat32(b));
    return *this;
  }

  Float8_e4m3fn &operator*=(const Float8_e4m3fn &b) {
    value_ = FromFloat32(ToFloat32(*this) * ToFloat32(b));
    return *this;
  }

  Float8_e4m3fn &operator/=(const Float8_e4m3fn &b) {
    value_ = FromFloat32(ToFloat32(*this) / ToFloat32(b));
    return *this;
  }

  static float ToFloat32(const Float8_e4m3fn &e4m3fn) {
    constexpr uint32_t mu_value = 121 << 23;
    constexpr Union32 magic{mu_value};
    constexpr uint32_t exponent_adjust = ((127 - 7) << 23);
    constexpr uint32_t nan_extra_exp_adjust = ((128 - 8) << 23);
    constexpr uint32_t zero_extra_exp_adjust = (1 << 23);
    constexpr uint32_t sign_mask = 0x80;
    constexpr unsigned int shifted_exp = (0x3c00 << 13);
    constexpr unsigned int nan_shifted_exp = (0x3f80 << 13);
    constexpr unsigned int exponent_bits = 20;
    constexpr unsigned int sign_bit_shift = 24;
    // Exponent/mantissa bits.
    Union32 f32;
    f32.u = (static_cast<uint32_t>(e4m3fn.value_ & value_mask) << exponent_bits);
    // Just the exponent.
    unsigned int exp = (shifted_exp & f32.u);
    bool is_nan = ((nan_shifted_exp & f32.u) == nan_shifted_exp);
    f32.u += exponent_adjust;
    // Handle exponent special cases.
    if (is_nan) {
      // Inf/NaN, extra exp adjust.
      f32.u += nan_extra_exp_adjust;
    } else if (exp == 0) {
      // Zero/Denormal, extra exp adjust and renormalize.
      f32.u += zero_extra_exp_adjust;
      f32.f -= magic.f;
    }
    // Set sign bit.
    f32.u |= ((e4m3fn.value_ & sign_mask) << sign_bit_shift);
    return f32.f;
  }

 private:
  static uint8_t FromFloat32(float f32) {
    constexpr uint32_t magic = {121 << 23};
    constexpr uint32_t e4m3max_value = 1087 << 20;
    constexpr Union32 e4m3max{e4m3max_value};
    constexpr uint32_t denorm_magic_value = ((127 - 7) + (23 - 3) + 1) << 23;
    constexpr Union32 denorm_magic{denorm_magic_value};
    constexpr unsigned int exponent_bits = 20;
    constexpr unsigned int sign_bit_shift = 24;
    constexpr unsigned int sign_mask = 0x80000000u;
    constexpr uint32_t rouding_bias_part1 = (static_cast<unsigned int>(7 - 127) << 23) + 0x7ffff;

    Union32 f;
    f.f = f32;
    unsigned int sign = f.u & sign_mask;
    f.u ^= sign;
    uint8_t result = 0;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).
    if (f.u >= e4m3max.u) {
      // Result is NaN .
      // Attention that FP8 E4M3FN format does not support representation of infinity (INF).
      result = nan_value;
    } else if (f.u < magic) {
      // (De)normalized number or zero; resulting float8_e4m3fn is subnormal or zero.
      // Use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;
      // And one integer subtract of the bias later, we have our final float!
      result = static_cast<uint8_t>(f.u - denorm_magic.u);
    } else {
      // Resulting mantissa is odd.
      unsigned int mant_odd = (f.u >> exponent_bits) & 1;
      // Update exponent, rounding bias part 1;
      f.u += rouding_bias_part1;
      // Rounding bias part 2;
      f.u += mant_odd;
      // Take the bits!
      result = static_cast<uint8_t>(f.u >> exponent_bits);
    }
    // Set sign bit.
    result |= static_cast<uint8_t>(sign >> sign_bit_shift);
    return result;
  }

  uint8_t value_;
};

inline Float8_e4m3fn operator+(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return Float8_e4m3fn(static_cast<float>(a) + static_cast<float>(b));
}

inline Float8_e4m3fn operator*(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return Float8_e4m3fn(static_cast<float>(a) * static_cast<float>(b));
}

inline Float8_e4m3fn operator-(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return Float8_e4m3fn(static_cast<float>(a) - static_cast<float>(b));
}

inline Float8_e4m3fn operator/(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return Float8_e4m3fn(static_cast<float>(a) / static_cast<float>(b));
}

// Division by an size_t. Do it in full float precision to avoid
// accuracy issues in converting the denominator to Float8_e4m3fn.
inline Float8_e4m3fn operator/(const Float8_e4m3fn &a, size_t b) {
  return Float8_e4m3fn(static_cast<float>(a) / static_cast<float>(b));
}

inline Float8_e4m3fn operator-(const Float8_e4m3fn &a) {
  constexpr uint8_t sign_mask = 0x80;
  return Float8_e4m3fn::FromRaw(a.int_value() ^ sign_mask);
}

inline bool operator==(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return std::equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator!=(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return std::not_equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator<(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
inline bool operator<=(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator>(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator>=(const Float8_e4m3fn &a, const Float8_e4m3fn &b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

inline std::ostream &operator<<(std::ostream &os, const Float8_e4m3fn &v) { return (os << static_cast<float>(v)); }

}  // namespace mindspore

using float8_e4m3fn = mindspore::Float8_e4m3fn;

namespace std {
template <>
struct hash<float8_e4m3fn> {
  std::size_t operator()(const float8_e4m3fn &fp8_e4m3fn) const noexcept {
    return static_cast<std::size_t>(fp8_e4m3fn.int_value());
  }
};

template <>
struct is_floating_point<float8_e4m3fn> : public std::true_type {};

template <>
struct is_signed<float8_e4m3fn> : public std::true_type {};

// If std::numeric_limits<T> is specialized, should also specialize
// std::numeric_limits<const T>, std::numeric_limits<volatile T>, and
// std::numeric_limits<const volatile T>
// https://stackoverflow.com/a/16519653/
template <>
struct numeric_limits<const mindspore::Float8_e4m3fn> : private numeric_limits<mindspore::Float8_e4m3fn> {};
template <>
struct numeric_limits<volatile mindspore::Float8_e4m3fn> : private numeric_limits<mindspore::Float8_e4m3fn> {};
template <>
struct numeric_limits<const volatile mindspore::Float8_e4m3fn> : private numeric_limits<mindspore::Float8_e4m3fn> {};
}  // namespace std

#endif  // MINDSPORE_CORE_BASE_FLOAT8_E4M3FN_H_
