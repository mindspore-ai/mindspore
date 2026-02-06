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

#include "backend/ms_backend/runtime/graph_executor/pipeline/async_lf_queue.h"
#include <future>
#include "common/common_test.h"

namespace mindspore {
namespace runtime {
class AsyncLFQueueTest : public UT::Common {
 public:
  AsyncLFQueueTest() {}
};

/// Feature: Test single producer single consumer case for async lock free queue naive function.
/// Description: Single producer single consumer case, push and pop must behave as correct execution order.
/// Expectation: The result is expected.
TEST_F(AsyncLFQueueTest, TaskExecutionOrder) {
  AsyncLFQueue queue("TaskExecutionOrder");
  queue.Init();
  queue.Continue();
  std::vector<int> results;
  std::mutex mtx;

  for (int i = 1; i <= 3; ++i) {
    queue.Push([i, &results, &mtx]() {
      std::lock_guard<std::mutex> lock(mtx);
      results.push_back(i);
    });
  }

  queue.Wait();

  ASSERT_EQ(results.size(), 3);
  EXPECT_EQ(results[0], 1);
  EXPECT_EQ(results[1], 2);
  EXPECT_EQ(results[2], 3);
}

/// Feature: Test multi producer single consumer case for async lock free queue.
/// Description: Multi producer single consumer case, push and pop must behave as correct execution order.
/// Expectation: The result is expected.
TEST_F(AsyncLFQueueTest, MultiProducerSingleConsumerExecutionOrder) {
  AsyncLFQueue queue("MPSCOrderTest");
  queue.Init();
  queue.Continue();

  std::vector<int> execution_order;
  std::mutex order_mutex;
  std::atomic<int> task_count{0};
  constexpr int kTotalTasks = 500;

  std::vector<std::thread> producers;
  for (int i = 0; i < 3; ++i) {
    producers.emplace_back([&] {
      for (int j = 0; j < kTotalTasks; ++j) {
        queue.Push([&task_count, &execution_order, &order_mutex] {
          int taskId = task_count++;
          std::lock_guard<std::mutex> lock(order_mutex);
          execution_order.push_back(taskId);
        });
      }
    });
  }

  for (auto &t : producers) t.join();
  queue.Wait();

  for (size_t i = 1; i < execution_order.size(); ++i) {
    EXPECT_LT(execution_order[i - 1], execution_order[i]) << "Task execution order violated at position " << i;
  }
}

/// Feature: Test single producer single consumer case for async lock free queue pause and continue.
/// Description: Single producer single consumer case, pause and continue function.
/// Expectation: The result is expected.
TEST_F(AsyncLFQueueTest, PauseContinueExecution) {
  AsyncLFQueue queue("PauseContinueTest");
  queue.Init();
  queue.Continue();
  std::atomic<int> counter{0};

  queue.Push([&counter]() { counter = 100; });
  queue.Wait();
  ASSERT_EQ(counter, 100);

  queue.Pause();
  EXPECT_THROW(queue.Push([&counter]() { counter = 200; }), std::exception);

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(counter, 100);

  queue.Continue();
  queue.Push([&counter]() { counter = 200; });
  queue.Wait();
  EXPECT_EQ(counter, 200);
}

/// Feature: Test single producer single consumer case for async lock free queue block wait.
/// Description: Single producer single consumer case, block wait function.
/// Expectation: The result is expected.
TEST_F(AsyncLFQueueTest, WaitCompletion) {
  AsyncLFQueue queue("WaitTest");
  queue.Init();
  queue.Continue();
  std::atomic<bool> task_started{false};
  std::atomic<bool> task_finished{false};

  queue.Push([&]() {
    task_started = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    task_finished = true;
  });

  while (!task_started) {
  }

  std::thread waiter([&]() {
    queue.Wait();
    EXPECT_TRUE(task_finished);
  });

  waiter.join();
  ASSERT_TRUE(task_finished);
}

/// Feature: Test single producer single consumer case for async lock free queue shutdown.
/// Description: Single producer single consumer case, shutdown function.
/// Expectation: The result is expected.
TEST_F(AsyncLFQueueTest, WorkerCleanShutdown) {
  std::promise<void> shutdown_promise;
  AsyncLFQueue *queue = new AsyncLFQueue("ShutdownTest");
  queue->Init();
  queue->Continue();

  queue->Push([&shutdown_promise]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    shutdown_promise.set_value();
  });

  auto future = shutdown_promise.get_future();
  future.wait();
  delete queue;
}
}  // namespace runtime
}  // namespace mindspore
