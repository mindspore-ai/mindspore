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
#ifndef MINDSPORE_CORE_BASE_HIFLOAT8_H_
#define MINDSPORE_CORE_BASE_HIFLOAT8_H_

#include <type_traits>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <limits>
#include <functional>
#include "mindapi/base/macros.h"

// Implement HiFloat8 for mindspore  https://arxiv.org/abs/2409.16626
namespace mindspore {
class MS_CORE_API HiFloat8 {
 public:
  static constexpr uint8_t value_mask = 0x7f;
  static constexpr uint8_t true_value = 0x08;
  static constexpr uint8_t inf_value = 0x6F;
  static constexpr uint8_t nan_value = 0x80;
  static constexpr uint8_t zero_value = 0x00;
  static constexpr uint32_t f32_inf_value = 0x7f800000;
  static constexpr uint32_t f32_zero_value = 0x00000000;
  static constexpr uint32_t f32_nan_value = 0x7fc00000;
  static constexpr uint32_t f32_value_mask = 0x7fffffff;
  static constexpr uint32_t hif8_sign_mask = 0x80;

  union Union32 {
    uint32_t u;
    float f;
  };

  HiFloat8() = default;
  ~HiFloat8() = default;

  HiFloat8(const HiFloat8 &other) noexcept = default;
  HiFloat8(HiFloat8 &&other) noexcept = default;

  HiFloat8 &operator=(const HiFloat8 &other) noexcept = default;
  HiFloat8 &operator=(HiFloat8 &&other) noexcept = default;

  static HiFloat8 FromRaw(uint8_t v) {
    HiFloat8 f;
    f.value_ = v;
    return f;
  }

  explicit HiFloat8(float f) : value_(FromFloat32(f)) {}
  explicit HiFloat8(bool b) : value_(b ? true_value : 0) {}
  template <typename T>
  explicit HiFloat8(const T &v) : value_(FromFloat32(static_cast<float>(v))) {}

  uint8_t int_value() const { return value_; }

  template <typename T>
  explicit operator T() const {
    return static_cast<T>(ToFloat32(*this));
  }

  explicit operator bool() const { return (value_ & value_mask) != 0; }
  explicit operator float() const { return ToFloat32(*this); }

  HiFloat8 &operator+=(const HiFloat8 &b) {
    value_ = FromFloat32(ToFloat32(*this) + ToFloat32(b));
    return *this;
  }

  HiFloat8 &operator-=(const HiFloat8 &b) {
    value_ = FromFloat32(ToFloat32(*this) - ToFloat32(b));
    return *this;
  }

  HiFloat8 &operator*=(const HiFloat8 &b) {
    value_ = FromFloat32(ToFloat32(*this) * ToFloat32(b));
    return *this;
  }

  HiFloat8 &operator/=(const HiFloat8 &b) {
    value_ = FromFloat32(ToFloat32(*this) / ToFloat32(b));
    return *this;
  }

  enum class ExponentRange {
    INVALID = -1,
    ZERO = 0,      // exponent <= -23 or exponent = 0
    INF = 1,       // exponent > 16
    DOT_0000 = 2,  // -22 <= exponent <= -16
    DOT_0001 = 3,  // exponent == 0
    DOT_001 = 4,   // exponent == ±1
    DOT_01 = 5,    // ±(2,3)
    DOT_10 = 6,    // ±(4-7)
    DOT_11 = 7     // ±(8-15)
  };

  static ExponentRange GetExponentRange(int32_t exponent) {
    constexpr int8_t kInfThreshold = 16;
    constexpr int8_t kZeroThreshold = -23;
    constexpr int8_t kDot0000Min = -22;
    constexpr int8_t kDot0000Max = -16;
    constexpr int8_t kDot01Threshold = 3;
    constexpr int8_t kDot10Threshold = 7;
    constexpr int8_t kDot11Threshold = 15;

    if (exponent >= kInfThreshold) {
      return ExponentRange::INF;
    }
    if (exponent <= kZeroThreshold) {
      return ExponentRange::ZERO;
    }
    if (exponent >= kDot0000Min && exponent <= kDot0000Max) {
      return ExponentRange::DOT_0000;
    }
    if (exponent == 0) {
      return ExponentRange::DOT_0001;
    }
    const int32_t exponent_abs = std::abs(exponent);
    if (exponent_abs <= 1) {
      return ExponentRange::DOT_001;
    } else if (exponent_abs <= kDot01Threshold) {
      return ExponentRange::DOT_01;
    } else if (exponent_abs <= kDot10Threshold) {
      return ExponentRange::DOT_10;
    } else if (exponent_abs <= kDot11Threshold) {
      return ExponentRange::DOT_11;
    }
    return ExponentRange::INVALID;
  }

  static float ToFloat32(const HiFloat8 &hif8) {
    static constexpr uint32_t f32_inf_value = 0x7f800000;
    static constexpr uint32_t f32_zero_value = 0x00000000;
    static constexpr uint32_t f32_nan_value = 0x7fc00000;
    static constexpr uint32_t hif8_sign_mask = 0x80;

    uint32_t sign = (hif8.value_ & hif8_sign_mask);
    constexpr uint8_t sign_bit_shift = 32 - 8;

    constexpr uint8_t fp32_exponent_adjust = 127;
    constexpr uint8_t fp8_DML_adjust = 23;
    constexpr uint8_t fp32_mantissa_width = 23;
    constexpr uint8_t mantissa_width = 3;

    Union32 f32 = {0};
    uint32_t exponent_val = 0;
    uint32_t mantissa_value = 0;
    uint8_t hif8_exponent_sign_mask = 0;
    uint32_t exponent_sign = 0;
    if ((hif8.value_ & 0x78) == 0x00) {
      if (hif8.value_ == nan_value) {  // NaN
        f32.u = f32_nan_value;
        return f32.f;
      }
      if (hif8.value_ == zero_value) {  // zero
        f32.u = f32_zero_value;
        return f32.f;
      }
      // For DML, HiF8 should be interpreted as: X = (−1)^S × 2^(M−23) × 1.0
      exponent_val = (hif8.value_ & ((1U << mantissa_width) - 1)) + fp32_exponent_adjust - fp8_DML_adjust;
      f32.u = (static_cast<uint32_t>(exponent_val) << fp32_mantissa_width);
    } else if ((hif8.value_ & 0x78) == 0x08) {  // dot == 0001
      // For the normal number, HiF8 should be interpreted as: X = (−1)^S × 2^E × 1.M
      exponent_val = 0 + fp32_exponent_adjust;
      f32.u = (static_cast<uint32_t>(exponent_val) << fp32_mantissa_width);
      mantissa_value = hif8.value_ & ((1U << mantissa_width) - 1);          // 小数位为最后后三位
      f32.u |= (mantissa_value << (fp32_mantissa_width - mantissa_width));  // 加上小数位
    } else if ((hif8.value_ & 0x70) == 0x10) {                              // dot == 001
      hif8_exponent_sign_mask = 0x08;
      exponent_sign = ((hif8.value_ & hif8_exponent_sign_mask) >> mantissa_width);
      exponent_val = ((exponent_sign == 1) ? -1 : 1) + fp32_exponent_adjust;
      f32.u = (static_cast<uint32_t>(exponent_val) << fp32_mantissa_width);
      mantissa_value = hif8.value_ & ((1U << mantissa_width) - 1);
      f32.u |= (mantissa_value << (fp32_mantissa_width - mantissa_width));
    } else if (((hif8.value_ & 0x60) == 0x60) || ((hif8.value_ & 0x60) == 0x40) ||
               ((hif8.value_ & 0x60) == 0x20)) {      // dot == 11,10,01
      if ((hif8.value_ & value_mask) == inf_value) {  // Inf
        f32.u = f32_inf_value | (sign << sign_bit_shift);
        return f32.f;
      }
      constexpr uint8_t hif8_dot_mask = 0x60;
      constexpr uint8_t exp_man_width = 5;
      uint8_t exponent_width = ((hif8.value_ & hif8_dot_mask) >> exp_man_width) + 1;
      uint8_t mantissa_width_ = exp_man_width - exponent_width;
      hif8_exponent_sign_mask = 0x10;
      exponent_sign = ((hif8.value_ & hif8_exponent_sign_mask) >> (exp_man_width - 1));
      uint8_t hif8_exponent_mantissa_mask = 0x0F;
      exponent_val =
        ((exponent_sign == 1) ? -1 : 1) *
          ((1U << (exponent_width - 1)) + ((hif8.value_ & hif8_exponent_mantissa_mask) >> mantissa_width_)) +
        fp32_exponent_adjust;
      f32.u = (static_cast<uint32_t>(exponent_val) << fp32_mantissa_width);
      mantissa_value = hif8.value_ & ((1U << mantissa_width_) - 1);
      f32.u |= (mantissa_value << (fp32_mantissa_width - mantissa_width_));
    }

    f32.u |= (sign << sign_bit_shift);
    return f32.f;
  }

 private:
  static uint8_t FromFloat32(float f32) {
    constexpr uint32_t f32infty_value = 255 << 23;
    constexpr Union32 f32infty{f32infty_value};
    constexpr uint32_t f8max_value = (127 + 15) << 23;
    constexpr Union32 f8max{f8max_value};
    constexpr int8_t fp8_DML_adjust = 23;
    Union32 f;
    f.f = f32;

    if ((f.u & f32_value_mask) == f32_zero_value) {
      return zero_value;
    }

    constexpr unsigned int sign_mask = 0x80000000u;
    unsigned int sign = f.u & sign_mask;
    uint32_t sign_bits = ((f.u >> 31) & 1) ? 0x80 : 0x00;
    f.u ^= sign;

    if (f.u > f8max.u) {
      // Result is Inf or NaN (all exponent bits set).
      return (f.u > f32infty.u) ? nan_value : (sign_bits | inf_value);
    }

    int32_t exponent = ((f.u >> 23) & 0xFF) - 127;  // 去除偏置
    uint32_t mantissa = f.u & 0x7FFFFF;

    ExponentRange range = GetExponentRange(exponent);

    switch (range) {
      case ExponentRange::ZERO:
        return zero_value;

      case ExponentRange::INF:
        return sign_bits | inf_value;

      case ExponentRange::DOT_0000: {
        exponent += fp8_DML_adjust;
        if (exponent > std::numeric_limits<uint8_t>::max()) {
          MS_LOG(INTERNAL_EXCEPTION) << "The exponent " << exponent << " exceeds the maximum value of uint8_t.";
        }
        uint8_t dot_bit = 0x00;
        int32_t mantissa_width = 3;
        int32_t exponent_width = 0;
        return (sign_bits | (dot_bit << (mantissa_width + exponent_width)) | (uint32_t)exponent);
      }

      case ExponentRange::DOT_0001: {
        uint8_t dot_bit = 0b0001;
        int32_t mantissa_width = 3;
        int32_t exponent_width = 0;
        return (sign_bits | (dot_bit << (mantissa_width + exponent_width)) |
                (mantissa >> (fp8_DML_adjust - mantissa_width)));
      }

      case ExponentRange::DOT_001: {
        uint8_t dot_bit = 0b001;
        uint8_t exponent_bit = ((exponent > 0) ? 0 : 1);
        int32_t mantissa_width = 3;
        int32_t exponent_width = 1;
        return (sign_bits | (dot_bit << (mantissa_width + exponent_width)) | (exponent_bit << mantissa_width) |
                (mantissa >> (fp8_DML_adjust - mantissa_width)));
      }

      case ExponentRange::DOT_01: {
        uint8_t dot_bit = 0b01;
        uint8_t exponent_bit = ((exponent > 0) ? ((uint32_t)exponent & 0x1) : (uint32_t)(-exponent));
        int32_t mantissa_width = 3;
        int32_t exponent_width = 2;
        return (sign_bits | (dot_bit << (mantissa_width + exponent_width)) | (exponent_bit << mantissa_width) |
                (mantissa >> (fp8_DML_adjust - mantissa_width)));
      }

      case ExponentRange::DOT_10: {
        uint8_t dot_bit = 0b10;
        uint8_t exponent_bit = ((exponent > 0) ? ((uint32_t)exponent & 0x3) : (uint32_t)(-exponent));
        int32_t mantissa_width = 2;
        int32_t exponent_width = 3;
        return (sign_bits | (dot_bit << (mantissa_width + exponent_width)) | (exponent_bit << mantissa_width) |
                (mantissa >> (fp8_DML_adjust - mantissa_width)));
      }

      case ExponentRange::DOT_11: {
        uint8_t dot_bit = 0b11;
        uint8_t exponent_bit = ((exponent > 0) ? ((uint32_t)exponent & 0x7) : (uint32_t)(-exponent));
        int32_t mantissa_width = 1;
        int32_t exponent_width = 4;
        return (sign_bits | (dot_bit << (mantissa_width + exponent_width)) | (exponent_bit << mantissa_width) |
                (mantissa >> (fp8_DML_adjust - mantissa_width)));
      }

      default:
        return 0x00;
    }
  }
  uint8_t value_;
};

inline HiFloat8 operator+(const HiFloat8 &a, const HiFloat8 &b) {
  return HiFloat8(static_cast<float>(a) + static_cast<float>(b));
}

inline HiFloat8 operator*(const HiFloat8 &a, const HiFloat8 &b) {
  return HiFloat8(static_cast<float>(a) * static_cast<float>(b));
}

inline HiFloat8 operator-(const HiFloat8 &a, const HiFloat8 &b) {
  return HiFloat8(static_cast<float>(a) - static_cast<float>(b));
}

inline HiFloat8 operator/(const HiFloat8 &a, const HiFloat8 &b) {
  return HiFloat8(static_cast<float>(a) / static_cast<float>(b));
}

// Division by an size_t. Do it in full float precision to avoid
// accuracy issues in converting the denominator to hifloat8.
inline HiFloat8 operator/(const HiFloat8 &a, size_t b) {
  return HiFloat8(static_cast<float>(a) / static_cast<float>(b));
}

inline HiFloat8 operator-(const HiFloat8 &a) {
  constexpr uint8_t sign_mask = 0x80;
  return HiFloat8::FromRaw(a.int_value() ^ sign_mask);
}

inline bool operator==(const HiFloat8 &a, const HiFloat8 &b) {
  return std::equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator!=(const HiFloat8 &a, const HiFloat8 &b) {
  return std::not_equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator<(const HiFloat8 &a, const HiFloat8 &b) { return static_cast<float>(a) < static_cast<float>(b); }
inline bool operator<=(const HiFloat8 &a, const HiFloat8 &b) { return static_cast<float>(a) <= static_cast<float>(b); }
inline bool operator>(const HiFloat8 &a, const HiFloat8 &b) { return static_cast<float>(a) > static_cast<float>(b); }
inline bool operator>=(const HiFloat8 &a, const HiFloat8 &b) { return static_cast<float>(a) >= static_cast<float>(b); }

inline std::ostream &operator<<(std::ostream &os, const HiFloat8 &v) { return (os << static_cast<float>(v)); }

}  // namespace mindspore

using hifloat8 = mindspore::HiFloat8;

namespace std {
template <>
struct hash<hifloat8> {
  std::size_t operator()(const hifloat8 &hif8) const noexcept { return static_cast<std::size_t>(hif8.int_value()); }
};

template <>
struct is_floating_point<hifloat8> : public std::true_type {};

template <>
struct is_signed<hifloat8> : public std::true_type {};

// If std::numeric_limits<T> is specialized, should also specialize
// std::numeric_limits<const T>, std::numeric_limits<volatile T>, and
// std::numeric_limits<const volatile T>
// https://stackoverflow.com/a/16519653/
template <>
struct numeric_limits<const mindspore::HiFloat8> : private numeric_limits<mindspore::HiFloat8> {};
template <>
struct numeric_limits<volatile mindspore::HiFloat8> : private numeric_limits<mindspore::HiFloat8> {};
template <>
struct numeric_limits<const volatile mindspore::HiFloat8> : private numeric_limits<mindspore::HiFloat8> {};
}  // namespace std

#endif  // MINDSPORE_CORE_BASE_HIFLOAT8_H_
