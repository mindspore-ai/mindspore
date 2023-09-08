/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_BASE_BFLOAT16_H_
#define MINDSPORE_CORE_BASE_BFLOAT16_H_

#include <type_traits>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <limits>
#include <functional>
#include "third_party/securec/include/securec.h"

// Implement BFloat16 for mindspore, inspired by Eigen::half.
namespace mindspore {
class BFloat16 {
 public:
  static constexpr uint16_t value_mask = 0x7fff;
  static constexpr uint16_t inf_value = 0x7f80;
  static constexpr uint16_t nan_value = 0x7fc0;
  static constexpr uint16_t true_value = 0x3c00;
  static constexpr uint32_t f32_inf_value = 0x7f800000;

  BFloat16() = default;
  ~BFloat16() = default;

  BFloat16(const BFloat16 &other) noexcept = default;
  BFloat16(BFloat16 &&other) noexcept = default;

  BFloat16 &operator=(const BFloat16 &other) noexcept = default;
  BFloat16 &operator=(BFloat16 &&other) noexcept = default;

  static BFloat16 FromRaw(uint16_t v) {
    BFloat16 f;
    f.value_ = v;
    return f;
  }

  explicit BFloat16(float f) : value_(FromFloat32(f)) {}
  explicit BFloat16(bool b) : value_(b ? true_value : 0) {}
  template <typename T>
  explicit BFloat16(const T &v) : value_(FromFloat32(static_cast<float>(v))) {}

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

  BFloat16 &operator+=(const BFloat16 &b) {
    value_ = FromFloat32(ToFloat32(*this) + ToFloat32(b));
    return *this;
  }

  BFloat16 &operator-=(const BFloat16 &b) {
    value_ = FromFloat32(ToFloat32(*this) - ToFloat32(b));
    return *this;
  }

  BFloat16 &operator*=(const BFloat16 &b) {
    value_ = FromFloat32(ToFloat32(*this) * ToFloat32(b));
    return *this;
  }

  BFloat16 &operator/=(const BFloat16 &b) {
    value_ = FromFloat32(ToFloat32(*this) / ToFloat32(b));
    return *this;
  }

  static float ToFloat32(const BFloat16 &bf16) {
    // We should use memcpy in order to respect the strict aliasing rule.
    float f32 = 0;
    uint32_t f32_tmp = bf16.int_value();
    f32_tmp <<= 16;
    auto ret_code = memcpy_s(&f32, sizeof(f32), &f32_tmp, sizeof(f32_tmp));
    if (ret_code != 0) {
      return f32_inf_value;
    }
    return f32;
  }

 private:
  static uint16_t FromFloat32(float f32) {
    if (std::isnan(f32)) {
      return nan_value;
    } else {
      union {
        uint32_t U32;
        float F32;
      };
      F32 = f32;
      uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
      return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
  }

  uint16_t value_;
};

inline BFloat16 operator+(const BFloat16 &a, const BFloat16 &b) {
  return BFloat16(static_cast<float>(a) + static_cast<float>(b));
}

inline BFloat16 operator*(const BFloat16 &a, const BFloat16 &b) {
  return BFloat16(static_cast<float>(a) * static_cast<float>(b));
}

inline BFloat16 operator-(const BFloat16 &a, const BFloat16 &b) {
  return BFloat16(static_cast<float>(a) - static_cast<float>(b));
}

inline BFloat16 operator/(const BFloat16 &a, const BFloat16 &b) {
  return BFloat16(static_cast<float>(a) / static_cast<float>(b));
}

// Division by an size_t. Do it in full float precision to avoid
// accuracy issues in converting the denominator to bfloat16.
inline BFloat16 operator/(const BFloat16 &a, size_t b) {
  return BFloat16(static_cast<float>(a) / static_cast<float>(b));
}

inline BFloat16 operator-(const BFloat16 &a) {
  constexpr uint16_t sign_mask = 0x8000;
  return BFloat16::FromRaw(a.int_value() ^ sign_mask);
}

inline bool operator==(const BFloat16 &a, const BFloat16 &b) {
  return std::equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator!=(const BFloat16 &a, const BFloat16 &b) {
  return std::not_equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator<(const BFloat16 &a, const BFloat16 &b) { return static_cast<float>(a) < static_cast<float>(b); }
inline bool operator<=(const BFloat16 &a, const BFloat16 &b) { return static_cast<float>(a) <= static_cast<float>(b); }
inline bool operator>(const BFloat16 &a, const BFloat16 &b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator>=(const BFloat16 &a, const BFloat16 &b) { return static_cast<float>(a) >= static_cast<float>(b); }

inline std::ostream &operator<<(std::ostream &os, const BFloat16 &v) { return (os << static_cast<float>(v)); }

}  // namespace mindspore

using bfloat16 = mindspore::BFloat16;

namespace std {
template <>
struct hash<bfloat16> {
  std::size_t operator()(const bfloat16 &f16) const noexcept { return static_cast<std::size_t>(f16.int_value()); }
};

template <>
struct is_floating_point<bfloat16> : public std::true_type {};

template <>
struct is_signed<bfloat16> : public std::true_type {};

template <>
struct numeric_limits<bfloat16> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr std::float_denorm_style has_denorm = numeric_limits<float>::has_denorm;
  static constexpr bool has_denorm_loss = numeric_limits<float>::has_denorm_loss;
  static constexpr std::float_round_style round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr bool traps = numeric_limits<float>::traps;
  static constexpr bool tinyness_before = numeric_limits<float>::tinyness_before;

  static constexpr uint16_t raw_min = 0x0080;
  static constexpr uint16_t raw_max = 0x7f7f;
  static constexpr uint16_t raw_lowest = 0xff7f;
  static constexpr uint16_t raw_epsilon = 0x3c00;
  static constexpr uint16_t raw_round_error = 0x3f00;
  static constexpr uint16_t raw_infinity = 0x7f80;
  static constexpr uint16_t raw_quiet_nan = 0x7fc0;
  static constexpr uint16_t raw_signaling_nan = 0x7f80;
  static constexpr uint16_t raw_denorm_min = 0x0001;

  static bfloat16(min)() noexcept { return bfloat16::FromRaw(raw_min); }
  static bfloat16(max)() noexcept { return bfloat16::FromRaw(raw_max); }
  static bfloat16 lowest() noexcept { return bfloat16::FromRaw(raw_lowest); }
  static bfloat16 epsilon() noexcept { return bfloat16::FromRaw(raw_epsilon); }
  static bfloat16 round_error() noexcept { return bfloat16::FromRaw(raw_round_error); }
  static bfloat16 infinity() noexcept { return bfloat16::FromRaw(raw_infinity); }
  static bfloat16 quiet_NaN() noexcept { return bfloat16::FromRaw(raw_quiet_nan); }
  static bfloat16 signaling_NaN() noexcept { return bfloat16::FromRaw(raw_signaling_nan); }
  static bfloat16 denorm_min() noexcept { return bfloat16::FromRaw(raw_denorm_min); }
};

// If std::numeric_limits<T> is specialized, should also specialize
// std::numeric_limits<const T>, std::numeric_limits<volatile T>, and
// std::numeric_limits<const volatile T>
// https://stackoverflow.com/a/16519653/
template <>
struct numeric_limits<const mindspore::BFloat16> : private numeric_limits<mindspore::BFloat16> {};
template <>
struct numeric_limits<volatile mindspore::BFloat16> : private numeric_limits<mindspore::BFloat16> {};
template <>
struct numeric_limits<const volatile mindspore::BFloat16> : private numeric_limits<mindspore::BFloat16> {};
}  // namespace std

// Implements standard math functions for bfloat16.
inline bool(isinf)(const bfloat16 &a) { return (a.int_value() & bfloat16::value_mask) == bfloat16::inf_value; }
inline bool(isnan)(const bfloat16 &a) { return (a.int_value() & bfloat16::value_mask) > bfloat16::inf_value; }
inline bool(isfinite)(const bfloat16 &a) { return !(isinf(a)) && !(isnan(a)); }
inline bfloat16 abs(const bfloat16 &a) { return bfloat16::FromRaw(a.int_value() & bfloat16::value_mask); }
inline bfloat16 exp(const bfloat16 &a) { return bfloat16(::expf(static_cast<float>(a))); }
inline bfloat16 log(const bfloat16 &a) { return bfloat16(::logf(static_cast<float>(a))); }
inline bfloat16 log1p(const bfloat16 &a) { return bfloat16(::log1pf(static_cast<float>(a))); }
inline bfloat16 log10(const bfloat16 &a) { return bfloat16(::log10f(static_cast<float>(a))); }
inline bfloat16 sqrt(const bfloat16 &a) { return bfloat16(::sqrtf(static_cast<float>(a))); }
inline bfloat16 sin(const bfloat16 &a) { return bfloat16(::sinf(static_cast<float>(a))); }
inline bfloat16 cos(const bfloat16 &a) { return bfloat16(::cosf(static_cast<float>(a))); }
inline bfloat16 tan(const bfloat16 &a) { return bfloat16(::tanf(static_cast<float>(a))); }
inline bfloat16 tanh(const bfloat16 &a) { return bfloat16(::tanhf(static_cast<float>(a))); }
inline bfloat16 floor(const bfloat16 &a) { return bfloat16(::floorf(static_cast<float>(a))); }
inline bfloat16 ceil(const bfloat16 &a) { return bfloat16(::ceilf(static_cast<float>(a))); }
inline bfloat16(min)(const bfloat16 &a, const bfloat16 &b) { return b < a ? b : a; }
inline bfloat16(max)(const bfloat16 &a, const bfloat16 &b) { return a < b ? b : a; }
inline bfloat16 pow(const bfloat16 &a, const bfloat16 &b) {
  return bfloat16(::powf(static_cast<float>(a), static_cast<float>(b)));
}

inline float bfloat_to_float(const bfloat16 &h) { return static_cast<float>(h); }

#endif  // MINDSPORE_CORE_BASE_BFLOAT16_H_
