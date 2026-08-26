/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_BASE_FLOAT8_E8M0FNU_H_
#define MINDSPORE_CORE_BASE_FLOAT8_E8M0FNU_H_

#include <type_traits>
#include <cmath>
#include <climits>
#include <cstdint>
#include <ostream>
#include <limits>
#include <functional>
#include "mindapi/base/macros.h"

// Implement float8_e8m0fnu for mindspore
// E8M0FNU format from OCP MX spec (Section 5.4.1):
// Binary: eeeeeeee (8 exponent bits, no sign, no mantissa)
// NaN = 0xFF; no zero (0x00 = 2^-127); no infinity
// Unsigned (is_signed = false)
namespace mindspore {
class MS_CORE_API Float8_e8m0fnu {
 public:
  static constexpr uint8_t nan_value = 0xFF;
  static constexpr uint8_t true_value = 0x7F;  // 1.0 = 2^0 (bias 127)

  union Union32 {
    uint32_t u;
    float f;
  };

  Float8_e8m0fnu() = default;
  ~Float8_e8m0fnu() = default;

  Float8_e8m0fnu(const Float8_e8m0fnu &other) noexcept = default;
  Float8_e8m0fnu(Float8_e8m0fnu &&other) noexcept = default;

  Float8_e8m0fnu &operator=(const Float8_e8m0fnu &other) noexcept = default;
  Float8_e8m0fnu &operator=(Float8_e8m0fnu &&other) noexcept = default;

  static Float8_e8m0fnu FromRaw(uint8_t v) {
    Float8_e8m0fnu f;
    f.value_ = v;
    return f;
  }

  explicit Float8_e8m0fnu(float f) : value_(FromFloat32(f)) {}
  explicit Float8_e8m0fnu(bool b) : value_(b ? true_value : 0) {}
  template <typename T>
  explicit Float8_e8m0fnu(const T &v) : value_(FromFloat32(static_cast<float>(v))) {}

  uint8_t int_value() const { return value_; }

  template <typename T>
  explicit operator T() const {
    return static_cast<T>(ToFloat32(*this));
  }

  explicit operator bool() const { return value_ != nan_value; }
  explicit operator float() const { return ToFloat32(*this); }

  // No arithmetic operators: E8M0 is a block scaling factor, not a computation type.

  static float ToFloat32(const Float8_e8m0fnu &e8m0) {
    uint8_t x = e8m0.value_;

    // NaN
    if (x == nan_value) {
      Union32 nan_bits;
      nan_bits.u = 0x7F800001;  // float32 NaN
      return nan_bits.f;
    }

    // 0x00 maps to 2^-127 (not zero!)
    if (x == 0) {
      Union32 min_bits;
      min_bits.u = 0x00400000;  // 2^-127 as float32
      return min_bits.f;
    }

    // Normal: value = 2^(x - 127), reconstruct as float32
    Union32 f32;
    f32.u = static_cast<uint32_t>(x) << 23;
    return f32.f;
  }

 private:
  static uint8_t FromFloat32(float f32) {
    Union32 f;
    f.f = f32;

    // Extract exponent
    uint32_t exponent = (f.u >> 23) & 0xFF;

    // Negative values and NaN/Inf map to NaN
    if ((f.u & 0x80000000) != 0) {
      return nan_value;
    }

    // Float32 NaN or Inf -> e8m0 NaN
    if (exponent == 0xFF) {
      return nan_value;
    }

    // Round to nearest, ties to even using guard/round/sticky bits
    uint8_t g = (f.u & 0x400000) > 0;  // guard bit (bit 22)
    uint8_t r = (f.u & 0x200000) > 0;  // round bit (bit 21)
    uint8_t s = (f.u & 0x1FFFFF) > 0;  // sticky bits (bits 20..0)
    uint8_t lsb = exponent & 1;        // target exponent LSB for ties-to-even

    bool round_up = false;
    if (g == 1) {
      if ((r == 1) || (s == 1)) {
        round_up = true;
      } else if (lsb == 1) {
        round_up = true;
      }
    }

    if (round_up) {
      exponent++;
    }

    // Clamp to valid range [0, 254]; 255 = NaN
    if (exponent >= nan_value) {
      return nan_value;
    }

    return static_cast<uint8_t>(exponent);
  }

  uint8_t value_;
};

// No arithmetic operators (+, -, *, /): E8M0 is a block scaling factor, not a computation type.
// No unary operator- (unsigned type).

inline bool operator==(const Float8_e8m0fnu &a, const Float8_e8m0fnu &b) {
  return std::equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator!=(const Float8_e8m0fnu &a, const Float8_e8m0fnu &b) {
  return std::not_equal_to<float>()(static_cast<float>(a), static_cast<float>(b));
}

inline bool operator<(const Float8_e8m0fnu &a, const Float8_e8m0fnu &b) {
  return static_cast<float>(a) < static_cast<float>(b);
}
inline bool operator<=(const Float8_e8m0fnu &a, const Float8_e8m0fnu &b) {
  return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator>(const Float8_e8m0fnu &a, const Float8_e8m0fnu &b) {
  return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator>=(const Float8_e8m0fnu &a, const Float8_e8m0fnu &b) {
  return static_cast<float>(a) >= static_cast<float>(b);
}

inline std::ostream &operator<<(std::ostream &os, const Float8_e8m0fnu &v) { return (os << static_cast<float>(v)); }

}  // namespace mindspore

using float8_e8m0fnu = mindspore::Float8_e8m0fnu;

namespace std {
template <>
struct hash<float8_e8m0fnu> {
  std::size_t operator()(const float8_e8m0fnu &fp8_e8m0fnu) const noexcept {
    return static_cast<std::size_t>(fp8_e8m0fnu.int_value());
  }
};

template <>
struct is_floating_point<float8_e8m0fnu> : public std::true_type {};

template <>
struct is_signed<float8_e8m0fnu> : public std::false_type {};

template <>
struct numeric_limits<const mindspore::Float8_e8m0fnu> : private numeric_limits<mindspore::Float8_e8m0fnu> {};
template <>
struct numeric_limits<volatile mindspore::Float8_e8m0fnu> : private numeric_limits<mindspore::Float8_e8m0fnu> {};
template <>
struct numeric_limits<const volatile mindspore::Float8_e8m0fnu> : private numeric_limits<mindspore::Float8_e8m0fnu> {};
}  // namespace std

#endif  // MINDSPORE_CORE_BASE_FLOAT8_E8M0FNU_H_
