/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#define private public
#include "base/hifloat8.h"

namespace mindspore {

class TestHiFloat8 : public UT::Common {
 public:
  TestHiFloat8() {}
};

/// Feature: Data format conversion in HiFloat8.
/// Description: ZeroConversion between HiFloat8 and Float32.
/// Expectation: No exception.
TEST_F(TestHiFloat8, ZeroConversion) {
  HiFloat8 num = HiFloat8::FromRaw(HiFloat8::FromFloat32(0.0f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
}

/// Feature: Data format conversion in HiFloat8.
/// Description: OneConversion between HiFloat8 and Float32.
/// Expectation: No exception.
TEST_F(TestHiFloat8, OneConversion) {
  HiFloat8 num = HiFloat8::FromRaw(HiFloat8::FromFloat32(1.0f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
}

/// Feature: Data format conversion in HiFloat8.
/// Description: NegativeOneConversion between HiFloat8 and Float32.
/// Expectation: No exception.
TEST_F(TestHiFloat8, NegativeOneConversion) {
  HiFloat8 num = HiFloat8::FromRaw(HiFloat8::FromFloat32(-1.0f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
}

/// Feature: Data format conversion in HiFloat8.
/// Description: NumberConversion between HiFloat8 and Float32.
/// Expectation: No exception.
TEST_F(TestHiFloat8, NumberConversion) {
  HiFloat8 num = HiFloat8::FromRaw(HiFloat8::FromFloat32(0.5f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
  num = HiFloat8::FromRaw(HiFloat8::FromFloat32(-2.5f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
  num = HiFloat8::FromRaw(HiFloat8::FromFloat32(-8.0f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
  num = HiFloat8::FromRaw(HiFloat8::FromFloat32(96.0f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
  num = HiFloat8::FromRaw(HiFloat8::FromFloat32(768.0f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
  num = HiFloat8::FromRaw(HiFloat8::FromFloat32(-8192.0f));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
}

/// Feature: Data format conversion in HiFloat8.
/// Description: InfinityConversion between HiFloat8 and Float32.
/// Expectation: No exception.
TEST_F(TestHiFloat8, InfinityConversion) {
  float inf = std::numeric_limits<float>::infinity();
  HiFloat8 num = HiFloat8::FromRaw(HiFloat8::FromFloat32(inf));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
}

/// Feature: Data format conversion in HiFloat8.
/// Description: NaNConversion between HiFloat8 and Float32.
/// Expectation: No exception.
TEST_F(TestHiFloat8, NaNConversion) {
  float nan = std::numeric_limits<float>::quiet_NaN();
  HiFloat8 num = HiFloat8::FromRaw(HiFloat8::FromFloat32(nan));
  EXPECT_EQ(HiFloat8::FromFloat32(HiFloat8::ToFloat32(num)), num.int_value());
}

}  // namespace mindspore
