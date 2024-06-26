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

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "runtime/pipeline/ring_queue.h"

class RingQueueTest : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() { GlobalMockObject::verify(); }

  static void SetUpTestCase() {}

  static void TearDownTestCase() {}
};

namespace mindspore {
class TestA {};

/// Feature: Test PyNative RingQueue.
/// Description: Test Enqueue/Dequeue for RingQueue.
/// Expectation: Enqueue/Dequeue execute success.
TEST_F(RingQueueTest, TestRingQueue1) {
  auto queue1 = RingQueue<std::shared_ptr<TestA>, 16>();
  for (size_t i = 0; i < 15; ++i) {
    queue1.Enqueue(std::make_shared<TestA>());
  }

  for (size_t i = 0; i < 15; ++i) {
    auto element = queue1.Head();
    ASSERT_NE(element, nullptr);
    (void)queue1.Dequeue();
  }

  ASSERT_EQ(queue1.IsEmpty(), true);
}
} // namespace mindspore
