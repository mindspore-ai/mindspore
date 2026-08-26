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
#include <cmath>
#include <limits>
#include <sstream>
#include "common/common_test.h"
#define private public
#include "base/float8_e8m0fnu.h"

namespace mindspore {

class TestFloat8e8m0fnu : public UT::Common {
 public:
  TestFloat8e8m0fnu() {}
  /// Reinterpret uint32_t bits as float.
  static float BitsToFloat(uint32_t bits) {
    Float8_e8m0fnu::Union32 f{bits};
    return f.f;
  }

  /// Reference RNE rounding for e8m0, matching OCP MX spec.
  static uint32_t RoundE8M0RNE(uint32_t biased_exp, uint8_t lsb, uint8_t g, uint8_t r, uint8_t s) {
    bool round_up = false;
    if (g == 1) {
      if ((r == 1) || (s == 1)) {
        round_up = true;
      } else if (lsb == 1) {
        round_up = true;
      }
    }
    if (round_up) {
      biased_exp++;
    }
    return biased_exp;
  }
};

// ======================== Conversion: special values ========================

/// Feature: Data format conversion in Float8_e8m0fnu.
/// Description: Table-driven verification of all special values against OCP MX spec.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, SpecialNumbers) {
  struct TestCase {
    uint8_t raw;
    float expected;
    const char *name;
  };
  // OCP MX spec Section 5.4.1.
  TestCase cases[] = {
    {0x00, std::ldexp(1.0f, -127), "smallest (2^-127)"},
    {0xFE, std::ldexp(1.0f, 127), "largest (2^127)"},
    {0x7E, 0.5f, "0.5 (2^-1)"},
    {0x7F, 1.0f, "1.0 (2^0)"},
    {0x80, 2.0f, "2.0 (2^1)"},
    {0x89, 1024.0f, "1024 (2^10)"},
  };
  for (const auto &c : cases) {
    Float8_e8m0fnu val = Float8_e8m0fnu::FromRaw(c.raw);
    float actual = Float8_e8m0fnu::ToFloat32(val);
    EXPECT_FLOAT_EQ(actual, c.expected) << "case: " << c.name;
  }

  // NaN (0xFF) tested separately: EXPECT_FLOAT_EQ cannot compare NaN.
  Float8_e8m0fnu nan_val = Float8_e8m0fnu::FromRaw(0xFF);
  EXPECT_TRUE(std::isnan(Float8_e8m0fnu::ToFloat32(nan_val)));
}

// ======================== Conversion: RNE rounding ========================

/// Feature: Data format conversion in Float8_e8m0fnu.
/// Description: Exhaustive RNE rounding: 256 exponents x 8 g/r/s = 2048 cases.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, RNERoundingExhaustive) {
  // For every possible e8m0 exponent (0..255) and every guard/round/sticky
  // combination (0..7), verify that RNE rounding is correctly applied when
  // casting from float32 to e8m0.
  // Reference: OCP MX spec Section 5.4.1, RNE rounding rules.
  for (uint32_t biased_exp = 0; biased_exp < 256; biased_exp++) {
    for (uint32_t grs = 0; grs < 8; grs++) {
      // Construct a float32 with the given exponent and g/r/s mantissa bits.
      uint32_t start_bits = (biased_exp << 23) | (grs << 20);
      float fp32_start = TestFloat8e8m0fnu::BitsToFloat(start_bits);

      // Compute expected exponent after RNE rounding.
      uint32_t expected_exp;
      if (biased_exp == 255) {
        expected_exp = biased_exp;
      } else {
        uint8_t lsb = biased_exp & 1;
        uint8_t g = (grs >> 2) & 1;
        uint8_t r = (grs >> 1) & 1;
        uint8_t s = grs & 1;
        expected_exp = TestFloat8e8m0fnu::RoundE8M0RNE(biased_exp, lsb, g, r, s);
      }

      // Determine expected float32 result after round-trip.
      bool expect_nan = expected_exp >= Float8_e8m0fnu::nan_value;

      // Actual: float32 -> e8m0 -> float32.
      Float8_e8m0fnu e8m0(fp32_start);
      float actual = Float8_e8m0fnu::ToFloat32(e8m0);

      if (expect_nan) {
        EXPECT_TRUE(std::isnan(actual)) << "exp=" << biased_exp << " grs=" << grs << " expected NaN, got " << actual;
      } else if (expected_exp == 0) {
        // e8m0 raw=0 maps to 2^-127 (0x00400000), not float32 zero.
        EXPECT_FLOAT_EQ(actual, TestFloat8e8m0fnu::BitsToFloat(0x00400000)) << "exp=" << biased_exp << " grs=" << grs;
      } else {
        float expected = TestFloat8e8m0fnu::BitsToFloat(expected_exp << 23);
        EXPECT_FLOAT_EQ(actual, expected) << "exp=" << biased_exp << " grs=" << grs;
      }
    }
  }
}

// ======================== Conversion: round-trip ========================

/// Feature: Data format conversion in Float8_e8m0fnu.
/// Description: Round-trip float32 -> e8m0 -> float32 for all 255 representable values.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, RoundTripPowersOfTwo) {
  // Every representable e8m0 value is an exact power of two: 2^(raw - 127).
  // Round-trip must be exact for all raw in [0, 254].
  for (int raw = 0; raw <= 254; raw++) {
    float expected = std::ldexp(1.0f, raw - 127);
    Float8_e8m0fnu e8m0(expected);
    float actual = Float8_e8m0fnu::ToFloat32(e8m0);
    EXPECT_FLOAT_EQ(actual, expected) << "raw=" << raw;
  }
}

// ======================== Conversion: edge cases ========================

/// Feature: Data format conversion in Float8_e8m0fnu.
/// Description: Zero creation clamps to 2^-127 (no zero in this format).
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, ZeroCreation) {
  // E8M0 has no zero representation; 0.0f gets clamped to 2^-127.
  Float8_e8m0fnu from_zero(0.0f);
  float val = Float8_e8m0fnu::ToFloat32(from_zero);
  EXPECT_GT(val, 0.0f);
  EXPECT_NEAR(val, std::ldexp(1.0f, -127), 1e-45f);
}

/// Feature: Data format conversion in Float8_e8m0fnu.
/// Description: Negative inputs map to NaN (unsigned type, no sign bit).
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, NegativeInput) {
  float neg_values[] = {-0.0f, -1.0f, -0.5f, -1024.0f, -std::numeric_limits<float>::infinity()};
  for (float v : neg_values) {
    Float8_e8m0fnu e8m0(v);
    EXPECT_EQ(e8m0.int_value(), 0xFF) << v << " should map to NaN";
    EXPECT_TRUE(std::isnan(Float8_e8m0fnu::ToFloat32(e8m0)));
  }
}

/// Feature: Data format conversion in Float8_e8m0fnu.
/// Description: Float32 inf and NaN inputs map to e8m0 NaN (0xFF).
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, InfAndNaNInput) {
  Float8_e8m0fnu from_inf(std::numeric_limits<float>::infinity());
  EXPECT_EQ(from_inf.int_value(), 0xFF);

  Float8_e8m0fnu from_qnan(std::numeric_limits<float>::quiet_NaN());
  EXPECT_EQ(from_qnan.int_value(), 0xFF);

  Float8_e8m0fnu from_snan(std::numeric_limits<float>::signaling_NaN());
  EXPECT_EQ(from_snan.int_value(), 0xFF);
}

// ======================== Operators: bool ========================

/// Feature: operator bool for Float8_e8m0fnu.
/// Description: Only NaN (0xFF) is falsy; all 255 finite values are truthy.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, OperatorBool) {
  EXPECT_FALSE(static_cast<bool>(Float8_e8m0fnu::FromRaw(0xFF)));

  for (int raw = 0; raw <= 254; raw++) {
    EXPECT_TRUE(static_cast<bool>(Float8_e8m0fnu::FromRaw(static_cast<uint8_t>(raw))))
      << "raw=" << raw << " should be truthy";
  }
}

// ======================== Operators: comparison ========================

/// Feature: Comparison operators for Float8_e8m0fnu.
/// Description: ==, !=, <, >, <=, >= between e8m0 values.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, Comparison) {
  Float8_e8m0fnu one(1.0f);
  Float8_e8m0fnu two(2.0f);
  Float8_e8m0fnu one_dup(1.0f);

  EXPECT_TRUE(one == one_dup);
  EXPECT_TRUE(one != two);
  EXPECT_TRUE(one < two);
  EXPECT_TRUE(two > one);
  EXPECT_TRUE(one <= one_dup);
  EXPECT_TRUE(one <= two);
  EXPECT_TRUE(two >= one);
  EXPECT_TRUE(one >= one_dup);
}

// ======================== Operators: ostream ========================

/// Feature: ostream operator<< for Float8_e8m0fnu.
/// Description: operator<< produces valid output.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, OStream) {
  std::ostringstream oss;
  oss << Float8_e8m0fnu(1.0f);
  EXPECT_FALSE(oss.str().empty());
}

// ======================== Type traits and hash ========================

/// Feature: std::is_floating_point and std::is_signed specialization.
/// Description: E8M0 is floating-point and unsigned.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, TypeTraits) {
  EXPECT_TRUE(std::is_floating_point<Float8_e8m0fnu>::value);
  EXPECT_FALSE(std::is_signed<Float8_e8m0fnu>::value);
}

/// Feature: std::hash specialization for Float8_e8m0fnu.
/// Description: Equal values produce equal hashes.
/// Expectation: No exception.
TEST_F(TestFloat8e8m0fnu, Hash) {
  std::hash<Float8_e8m0fnu> hasher;

  Float8_e8m0fnu a = Float8_e8m0fnu::FromRaw(0x7F);
  Float8_e8m0fnu b = Float8_e8m0fnu::FromRaw(0x7F);
  Float8_e8m0fnu c = Float8_e8m0fnu::FromRaw(0x80);

  EXPECT_EQ(hasher(a), hasher(b));
  EXPECT_NE(hasher(a), hasher(c));
}

}  // namespace mindspore
