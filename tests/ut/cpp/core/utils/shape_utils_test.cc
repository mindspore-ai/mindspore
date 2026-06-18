/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <stdexcept>
#include "common/common_test.h"
#include "utils/shape_utils.h"

namespace mindspore {
class TestShapeUtils : public UT::Common {
 public:
  TestShapeUtils() = default;
  void SetUp() {}
};

// Feature: NBytesOf
// Description: Check that NBytesOf returns the correct byte size for normal shapes and types.
// Expectation: The result equals item_size multiplied by element count.
TEST_F(TestShapeUtils, NBytesOfBasic) {
  ASSERT_EQ(NBytesOf(kNumberTypeFloat32, {2, 3}), 4 * 6);
  ASSERT_EQ(NBytesOf(kNumberTypeInt8, {2, 3, 4}), 1 * 24);
  ASSERT_EQ(NBytesOf(kNumberTypeFloat64, {1}), 8);
  ASSERT_EQ(NBytesOf(kNumberTypeComplex128, {2, 2}), 16 * 4);
  ASSERT_EQ(NBytesOf(kNumberTypeBool, {10}), 10);
  ASSERT_EQ(NBytesOf(kNumberTypeInt64, {4, 5}), 8 * 20);
  ASSERT_EQ(NBytesOf(kNumberTypeBFloat16, {3, 4}), 2 * 12);
}

// Feature: NBytesOf
// Description: Check that NBytesOf treats an empty shape (scalar) as a single element.
// Expectation: The result equals the item size of the data type.
TEST_F(TestShapeUtils, NBytesOfScalarShape) {
  ASSERT_EQ(NBytesOf(kNumberTypeFloat32, {}), 4);
  ASSERT_EQ(NBytesOf(kNumberTypeInt64, {}), 8);
  ASSERT_EQ(NBytesOf(kNumberTypeComplex128, {}), 16);
}

// Feature: NBytesOf
// Description: Check that NBytesOf returns zero for dynamic shapes that contain negative dimensions.
// Expectation: The result is zero because SizeOf returns zero for dynamic shapes.
TEST_F(TestShapeUtils, NBytesOfDynamicShape) {
  ASSERT_EQ(NBytesOf(kNumberTypeFloat32, {-1, 2}), 0);
  ASSERT_EQ(NBytesOf(kNumberTypeInt64, {-1}), 0);
  ASSERT_EQ(NBytesOf(kNumberTypeFloat32, {2, -1, 4}), 0);
}

// Feature: NBytesOf
// Description: Check that NBytesOf returns zero for an unknown data type whose item size is zero,
//              even when the shape is extremely large.
// Expectation: The result is zero and no overflow check is triggered.
TEST_F(TestShapeUtils, NBytesOfUnknownType) {
  ASSERT_EQ(NBytesOf(kTypeUnknown, {2, 3}), 0);
  ASSERT_EQ(NBytesOf(kTypeUnknown, {INT64_MAX}), 0);
}

// Feature: NBytesOf
// Description: Check that NBytesOf throws a runtime_error when item_size * element_count overflows size_t.
// Expectation: A std::runtime_error is thrown instead of returning a wrapped value.
TEST_F(TestShapeUtils, NBytesOfOverflow) {
  constexpr size_t kFloat32ItemSize = 4;
  int64_t overflow_dim = static_cast<int64_t>(SIZE_MAX / kFloat32ItemSize) + 1;
  EXPECT_THROW(NBytesOf(kNumberTypeFloat32, {overflow_dim}), std::runtime_error);

  constexpr size_t kComplex128ItemSize = 16;
  int64_t big_overflow_dim = static_cast<int64_t>(SIZE_MAX / kComplex128ItemSize) + 1;
  EXPECT_THROW(NBytesOf(kNumberTypeComplex128, {big_overflow_dim}), std::runtime_error);
}

// Feature: NBytesOf
// Description: Check the boundary value that does not overflow.
// Expectation: NBytesOf returns item_size * element_count for the largest non-overflowing element count.
TEST_F(TestShapeUtils, NBytesOfBoundaryNoOverflow) {
  constexpr size_t kFloat32ItemSize = 4;
  int64_t max_dim = static_cast<int64_t>(SIZE_MAX / kFloat32ItemSize);
  ASSERT_EQ(NBytesOf(kNumberTypeFloat32, {max_dim}), kFloat32ItemSize * static_cast<size_t>(max_dim));
}
}  // namespace mindspore
