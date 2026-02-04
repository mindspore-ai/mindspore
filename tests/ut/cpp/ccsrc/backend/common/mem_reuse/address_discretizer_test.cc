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

#include "common/common_test.h"
#include "runtime/memory/mem_pool/address_discretizer.h"
#include <gtest/gtest.h>

namespace mindspore {
namespace device {
namespace tracker {
namespace test {

class TestAddressDiscretizer : public UT::Common {};

/// Feature: AddressDiscretizer
/// Description: test basic functionality.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAddressDiscretizer, BasicFunctionality) {
  std::vector<uintptr_t> addresses = {0x3000, 0x1000, 0x4000, 0x2000};
  AddressDiscretizer discretizer(addresses);

  // Test GetDiscretizedCount
  EXPECT_EQ(discretizer.GetDiscretizedCount(), 4);

  // Test GetDiscreteId for existing addresses
  EXPECT_EQ(discretizer.GetDiscreteId(0x1000), 0);
  EXPECT_EQ(discretizer.GetDiscreteId(0x2000), 1);
  EXPECT_EQ(discretizer.GetDiscreteId(0x3000), 2);
  EXPECT_EQ(discretizer.GetDiscreteId(0x4000), 3);

  // Test GetDiscreteId for non-existing address
  EXPECT_EQ(discretizer.GetDiscreteId(0x5000), UINT32_MAX);

  // Test GetOriginalAddress
  EXPECT_EQ(discretizer.GetOriginalAddress(0), 0x1000);
  EXPECT_EQ(discretizer.GetOriginalAddress(1), 0x2000);
  EXPECT_EQ(discretizer.GetOriginalAddress(2), 0x3000);
  EXPECT_EQ(discretizer.GetOriginalAddress(3), 0x4000);

  // Test GetOriginalAddress for invalid id
  EXPECT_EQ(discretizer.GetOriginalAddress(4), 0);
  EXPECT_EQ(discretizer.GetOriginalAddress(UINT32_MAX), 0);
}

/// Feature: AddressDiscretizer
/// Description: test empty input.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAddressDiscretizer, EmptyInput) {
  std::vector<uintptr_t> empty_addresses;
  AddressDiscretizer empty_discretizer(empty_addresses);

  EXPECT_EQ(empty_discretizer.GetDiscretizedCount(), 0);
  EXPECT_EQ(empty_discretizer.GetDiscreteId(0x1000), UINT32_MAX);
  EXPECT_EQ(empty_discretizer.GetOriginalAddress(0), 0);
}

/// Feature: AddressDiscretizer
/// Description: test duplicate addresses.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAddressDiscretizer, DuplicateAddresses) {
  std::vector<uintptr_t> duplicate_addresses = {0x1000, 0x1000, 0x2000, 0x2000};
  AddressDiscretizer duplicate_discretizer(duplicate_addresses);

  EXPECT_EQ(duplicate_discretizer.GetDiscretizedCount(), 2);
  EXPECT_EQ(duplicate_discretizer.GetDiscreteId(0x1000), 0);
  EXPECT_EQ(duplicate_discretizer.GetDiscreteId(0x2000), 1);
}

/// Feature: AddressDiscretizer
/// Description: test large addresses.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestAddressDiscretizer, LargeAddresses) {
  std::vector<uintptr_t> large_addresses = {UINT64_MAX - 3, UINT64_MAX - 2, UINT64_MAX - 1, UINT64_MAX};
  AddressDiscretizer large_discretizer(large_addresses);

  EXPECT_EQ(large_discretizer.GetDiscretizedCount(), 4);
  EXPECT_EQ(large_discretizer.GetDiscreteId(UINT64_MAX - 3), 0);
  EXPECT_EQ(large_discretizer.GetDiscreteId(UINT64_MAX - 2), 1);
  EXPECT_EQ(large_discretizer.GetDiscreteId(UINT64_MAX - 1), 2);
  EXPECT_EQ(large_discretizer.GetDiscreteId(UINT64_MAX), 3);
}
}  // namespace test
}  // namespace tracker
}  // namespace device
}  // namespace mindspore