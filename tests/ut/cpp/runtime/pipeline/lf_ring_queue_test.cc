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

#include "backend/ms_backend/runtime/graph_executor/pipeline/lf_ring_queue.h"
#include <atomic>
#include <thread>
#include "common/common_test.h"

namespace mindspore {
namespace runtime {
class LFRingQueueTest : public UT::Common {
 public:
  LFRingQueueTest() {}
};

/// Feature: Test single producer single consumer case for lock free queue naive function.
/// Description: Single producer single consumer case, test Empty, Push, Front, Pop naive function.
/// Expectation: The result is expected.
TEST_F(LFRingQueueTest, BasicOperations) {
  LFRingQueue<int, 4> queue;
  queue.Continue();

  EXPECT_TRUE(queue.Empty());

  ASSERT_TRUE(queue.Push(1));
  ASSERT_FALSE(queue.Empty());

  EXPECT_EQ(*queue.Front(), 1);

  EXPECT_TRUE(queue.Pop());
  EXPECT_TRUE(queue.Empty());
}

/// Feature: Test single producer single consumer case for lock free queue blocking Front.
/// Description: Single producer single consumer case, test blocking Front function.
/// Expectation: The result is expected.
TEST_F(LFRingQueueTest, FrontBlockingBehavior) {
  LFRingQueue<int, 2> queue;
  queue.Continue();
  std::atomic<bool> started{false};

  std::thread consumer([&]() {
    started.store(true);
    auto *ptr = queue.Front();
    EXPECT_EQ(*ptr, 100);
  });

  while (!started) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  ASSERT_TRUE(queue.Push(100));
  consumer.join();
  ASSERT_TRUE(queue.Pop());
}

/// Feature: Test single producer single consumer case for lock free queue pause and continue.
/// Description: Single producer single consumer case, test pause and continue function.
/// Expectation: The result is expected.
TEST_F(LFRingQueueTest, PauseContinue) {
  LFRingQueue<int, 2> queue;
  queue.Continue();
  ASSERT_TRUE(queue.Push(10));

  queue.Pause();
  ASSERT_FALSE(queue.Push(20));

  queue.Continue();
  ASSERT_TRUE(queue.Push(30));

  EXPECT_EQ(*queue.Front(), 10);
  queue.Pop();
  EXPECT_EQ(*queue.Front(), 30);
}

/// Feature: Test multi producer single consumer case for lock free queue naive function.
/// Description: Multi producer single consumer case, test parallel push function.
/// Expectation: The result is expected.
TEST_F(LFRingQueueTest, MultiProducer) {
  LFRingQueue<int, 128> queue;
  queue.Continue();
  std::vector<std::thread> producers;
  const int itemCount = 1000;
  std::atomic<int> pushCount{0};

  for (int i = 0; i < 4; ++i) {
    producers.emplace_back([&]() {
      for (int j = 0; j < itemCount; ++j) {
        if (queue.Push(j)) pushCount++;
      }
    });
  }

  int popCount = 0;
  while (popCount < 4 * itemCount) {
    if (queue.Pop()) popCount++;
  }

  for (auto &t : producers) t.join();
  EXPECT_EQ(pushCount.load(), 4 * itemCount);
  EXPECT_EQ(popCount, 4 * itemCount);
  EXPECT_TRUE(queue.Empty());
}

/// Feature: Test single producer single consumer case for lock free queue full queue behavior.
/// Description: Single producer single consumer case, test full queue behavior.
/// Expectation: The result is expected.
TEST_F(LFRingQueueTest, FullQueueBehavior) {
  LFRingQueue<int, 2> queue;
  queue.Continue();

  ASSERT_TRUE(queue.Push(1));
  ASSERT_TRUE(queue.Push(2));

  std::thread producer([&]() { queue.Push(3); });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSERT_TRUE(queue.Pop());
  producer.join();
  EXPECT_EQ(*queue.Front(), 2);
}
}  // namespace runtime
}  // namespace mindspore
