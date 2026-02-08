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
#include "runtime/memory/mem_pool/race_checker.h"
#include <gtest/gtest.h>

namespace mindspore {
namespace device {
namespace tracker {
namespace graph {

class TestRaceChecker : public UT::Common {};

/// Feature: RaceChecker
/// Description: test read after write.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, ReadAfterWrite) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 1));
  EXPECT_TRUE(checker.CheckRead(0x1000, 0x2000, 0));
}

/// Feature: RaceChecker
/// Description: test write after write.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, WriteAfterWrite) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 1));
  EXPECT_TRUE(checker.CheckWrite(0x1000, 0x2000, 0));
}

/// Feature: RaceChecker
/// Description: test write after read.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, WriteAfterRead) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 1));
  EXPECT_TRUE(checker.CheckRead(0x1000, 0x2000, 0));
}

/// Feature: RaceChecker
/// Description: test read after read.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, ReadAfterRead) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckRead(0x1000, 0x2000, 1));
  EXPECT_FALSE(checker.CheckRead(0x1000, 0x2000, 0));
}

/// Feature: RaceChecker
/// Description: test safe read after write.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, SafeReadAfterWrite) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 1));
  checker.RecordEvent(1, "event");
  checker.WaitEvent(0, "event");
  EXPECT_FALSE(checker.CheckRead(0x1000, 0x2000, 0));
}

/// Feature: RaceChecker
/// Description: test safe write after write.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, SafeWriteAfterWrite) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 1));
  checker.RecordEvent(1, "event");
  checker.WaitEvent(0, "event");
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 0));
}

/// Feature: RaceChecker
/// Description: test safe write after read.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, SafeWriteAfterRead) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 1));
  checker.RecordEvent(1, "event");
  checker.WaitEvent(0, "event");
  EXPECT_FALSE(checker.CheckRead(0x1000, 0x2000, 0));
}

/// Feature: RaceChecker
/// Description: test overlap write after write.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, OverlapWriteAfterWrite) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 2);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x3000, 1));
  EXPECT_TRUE(checker.CheckWrite(0x2000, 0x4000, 0));
}

/// Feature: RaceChecker
/// Description: test cascade event.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestRaceChecker, CascadeEvent) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 3);
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 0));
  checker.RecordEvent(0, "event");
  checker.WaitEvent(1, "event");
  checker.RecordEvent(1, "event_1");
  checker.WaitEvent(2, "event_1");
  EXPECT_FALSE(checker.CheckWrite(0x1000, 0x2000, 2));
}

/// Feature: RaceChecker
/// Description: test illegal event.
/// Expectation: throw exception when input is invalid.
TEST_F(TestRaceChecker, IllegalEvent) {
  std::vector<uintptr_t> addresses = {0x1000, 0x2000, 0x3000, 0x4000};
  RaceChecker checker(addresses, 3);
  EXPECT_THROW(checker.WaitEvent(1, "event"), std::exception);
}

}  // namespace graph
}  // namespace tracker
}  // namespace device
}  // namespace mindspore
