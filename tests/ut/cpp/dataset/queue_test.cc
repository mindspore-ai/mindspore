/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "common/common.h"
#include "gtest/gtest.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/queue.h"
#include <atomic>
#include <chrono>
#include <random>
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestQueue : public UT::Common {
 public:
  MindDataTestQueue() {}

  void SetUp() {}
};

int gRefCountDestructorCalled;

class RefCount {
 public:
  RefCount() : v_(nullptr) {}
  explicit RefCount(int x) : v_(std::make_shared<int>(x)) {}
  RefCount(const RefCount &o) : v_(o.v_) {}
  ~RefCount() {
    MS_LOG(DEBUG) << "Destructor of RefCount called" << std::endl;
    gRefCountDestructorCalled++;
  }
  RefCount &operator=(const RefCount &o) {
    v_ = o.v_;
    return *this;
  }

  std::shared_ptr<int> v_;
};

TEST_F(MindDataTestQueue, Test1) {
  // Passing shared pointer along the queue
  Queue<std::shared_ptr<int>> que(3);
  std::shared_ptr<int> a = std::make_shared<int>(20);
  Status rc = que.Add(a);
  ASSERT_TRUE(rc.IsOk());
  // Use count should be 2 right now. a plus the one in the queue.
  ASSERT_EQ(a.use_count(), 2);
  std::shared_ptr<int> b;
  rc = que.PopFront(&b);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(*b, 20);
  // Use count should remain 2. a and b. No copy in the queue.
  ASSERT_EQ(a.use_count(), 2);
  a.reset(new int(5));
  ASSERT_EQ(a.use_count(), 1);
  // Push again but expect a is nullptr after push
  rc = que.Add(std::move(a));
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(a.use_count(), 0);
  rc = que.PopFront(&b);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(*b, 5);
  ASSERT_EQ(b.use_count(), 1);
  // Test construct in place
  rc = que.EmplaceBack(std::make_shared<int>(100));
  ASSERT_TRUE(rc.IsOk());
  rc = que.PopFront(&b);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(*b, 100);
  ASSERT_EQ(b.use_count(), 1);
  // Test the destructor of the Queue by add an element in the queue without popping it and let the queue go
  // out of scope.
  rc = que.EmplaceBack(std::make_shared<int>(2000));
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestQueue, Test2) {
  // Passing status object
  Queue<Status> que(3);
  Status rc_send(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Oops");
  Status rc = que.Add(rc_send);
  ASSERT_TRUE(rc.IsOk());
  Status rc_recv;
  rc = que.PopFront(&rc_recv);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(rc_recv, rc_send);
  rc = que.EmplaceBack(StatusCode::kOutOfMemory, "Test emplace");
  ASSERT_TRUE(rc.IsOk());
  Status rc_recv2;
  rc = que.PopFront(&rc_recv2);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_TRUE(rc_recv2.IsOutofMemory());
}

TEST_F(MindDataTestQueue, Test3) {
  Queue<std::unique_ptr<int>> que(3);
  std::unique_ptr<int> a(new int(3));
  Status rc = que.Add(std::move(a));
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(a.get(), nullptr);
  std::unique_ptr<int> b;
  rc = que.PopFront(&b);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(*b, 3);
  rc = que.EmplaceBack(new int(40));
  ASSERT_TRUE(rc.IsOk());
  rc = que.PopFront(&b);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(*b, 40);
}

void test4() {
  gRefCountDestructorCalled = 0;
  // Pass a structure along the queue.
  Queue<RefCount> que(3);
  RefCount a(3);
  Status rc = que.Add(a);
  ASSERT_TRUE(rc.IsOk());
  RefCount b;
  rc = que.PopFront(&b);
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(b.v_.use_count(), 2);
  ASSERT_EQ(*(b.v_.get()), 3);
  // Test the destructor of the Queue by adding an element without popping.
  rc = que.EmplaceBack(10);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestQueue, Test4) { test4(); }

TEST_F(MindDataTestQueue, Test5) {
  test4();
  // Assume we have run Test4. The destructor of the RefCount should be called 4 times.
  // One for a. One for b. One for line 125 when we pop. One for the stale element in the queue.
  ASSERT_EQ(gRefCountDestructorCalled, 4);
}

TEST_F(MindDataTestQueue, Test6) {
  // Create a list of queues
  QueueList<std::unique_ptr<int>> my_list_of_queues;
  const int chosen_queue_index = 2;
  const int num_queues = 4;
  const int queue_capacity = 3;
  my_list_of_queues.Init(num_queues, queue_capacity);
  // Now try to insert a number into a specific queue and pop it
  std::unique_ptr<int> a(new int(99));
  Status rc = my_list_of_queues[chosen_queue_index]->Add(std::move(a));
  ASSERT_TRUE(rc.IsOk());
  std::unique_ptr<int> pepped_value;
  rc = my_list_of_queues[chosen_queue_index]->PopFront(&pepped_value);
  ASSERT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "Popped value " << *pepped_value << " from queue index " << chosen_queue_index;
  ASSERT_EQ(*pepped_value, 99);
}
using namespace std::chrono;
template <typename QueueType, typename PayloadType>
void Perf(int n, int p, std::string name) {
  auto payload = std::vector<PayloadType>(n, PayloadType(p));
  auto queue = QueueType(n);
  auto t0 = high_resolution_clock::now();
  auto check = 0;
  for (int i = 0; i < queue.capacity(); i++) {
    queue.Add(PayloadType(p));
  }
  check = queue.size();
  for (int i = 0; i < queue.capacity(); i++) {
    queue.PopFront(&payload[i]);
  }
  auto t1 = high_resolution_clock::now();
  std::cout << name << " queue filled size: " << queue.size() << " " << check << std::endl;
  auto t2 = high_resolution_clock::now();
  for (int i = 0; i < queue.capacity(); i++) {
    queue.Add(PayloadType(p));
  }
  check = queue.size();
  for (int i = 0; i < queue.capacity(); i++) {
    queue.PopFront(&payload[i]);
  }
  auto t3 = high_resolution_clock::now();
  auto d = duration_cast<milliseconds>(t3 - t2 + t1 - t0).count();
  std::cout << name << " queue emptied size: " << queue.size() << " " << check << std::endl;
  std::cout << name << " "
            << " ran in " << d << "ms" << std::endl;
}

template <typename QueueType, typename PayloadType>
void Fuzz(int n, int p, std::string name) {
  std::mt19937 gen(1);
  auto payload = std::vector<PayloadType>(n, PayloadType(p));
  auto queue = QueueType(n);
  auto dist = std::uniform_int_distribution<int>(0, 2);
  std::cout << "###" << std::endl;
  for (auto i = 0; i < n; i++) {
    auto v = dist(gen);
    if (v == 0 && queue.size() < n - 1) {
      queue.Add(std::move(payload[i]));
    }
    if (v == 1 && queue.size() > 0) {
      queue.PopFront(&payload[i]);
    } else {
      queue.Reset();
    }
  }
  std::cout << name << " fuzz ran " << queue.size() << std::endl;
}
TEST_F(MindDataTestQueue, TestPerf) {
  try {
    int kSz = 1000000;
    //  std::cout << "enter size" << std::endl;
    //  std::cin >> kSz;
    Perf<Queue<std::vector<int>>, std::vector<int>>(kSz, 1, "old queue, vector of size 1");
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
  }

  std::cout << "Test Reset" << std::endl;
  std::cout << "Enter fuzz size" << std::endl;
  int fs = 1000;
//  std::cin >> fs;
  Fuzz<Queue<std::vector<int>>, std::vector<int>>(fs, 1, "New queue");
}
