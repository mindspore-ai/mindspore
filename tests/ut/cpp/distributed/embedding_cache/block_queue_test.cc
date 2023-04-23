/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <vector>
#include <thread>
#include <numeric>

#include "common/common_test.h"
#include "include/backend/distributed/embedding_cache/blocking_queue.h"

namespace mindspore {
namespace distributed {
class TestBlockingQueue : public UT::Common {
 public:
  TestBlockingQueue() = default;
  virtual ~TestBlockingQueue() = default;

  void SetUp() override {}
  void TearDown() override {}
};

void PushValue(std::vector<int> *push_values, BlockingQueue<int> *queue) {
  for (size_t i = 0; i < push_values->size(); ++i) {
    queue->Push(&(push_values->at(i)));
  }
}

void PopValue(size_t num, BlockingQueue<int> *queue, std::vector<int> *pop_values) {
  for (size_t i = 0; i < num; ++i) {
    pop_values->push_back(*(queue->Pop()));
  }
}

/// Feature: Test BlockingQueue all api.
/// Description: Test BlockingQueue data structure and interface.
/// Expectation: All interface work normally and results in line with expectations.
TEST_F(TestBlockingQueue, test_one_consumer_one_producer) {
  size_t capacity = 128;
  BlockingQueue<int> block_queue(capacity);
  EXPECT_EQ(block_queue.Empty(), true);
  EXPECT_EQ(block_queue.Full(), false);
  size_t ops_num = capacity * 10;
  std::vector<int> push_values(ops_num);
  std::iota(push_values.begin(), push_values.end(), 0);

  std::vector<int> pop_values;

  std::thread push_task(PushValue, &push_values, &block_queue);
  std::thread pop_task(PopValue, ops_num, &block_queue, &pop_values);

  push_task.join();
  pop_task.join();

  EXPECT_EQ(ops_num, pop_values.size());
  EXPECT_EQ(push_values, pop_values);
  EXPECT_NO_THROW(block_queue.Close());
  EXPECT_EQ(block_queue.Empty(), true);
  EXPECT_EQ(block_queue.Full(), false);
}
}  // namespace distributed
}  // namespace mindspore