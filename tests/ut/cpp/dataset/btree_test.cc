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

#include <sstream>
#include "minddata/dataset/util/btree.h"
#include "minddata/dataset/util/auto_index.h"
#include "minddata/dataset/util/system_pool.h"
#include "minddata/dataset/util/task_manager.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

// For testing purposes, we will make the branching factor very low.
struct mytraits {
  using slot_type = uint16_t;
  static const slot_type kLeafSlots = 6;
  static const slot_type kInnerSlots = 3;
};

class MindDataTestBPlusTree : public UT::Common {
 public:
  MindDataTestBPlusTree() = default;
};

// Test serial insert.
TEST_F(MindDataTestBPlusTree, Test1) {
  Allocator<std::string> alloc(std::make_shared<SystemPool>());
  BPlusTree<uint64_t, std::string, Allocator<std::string>, std::less<>, mytraits> btree(alloc);
  Status rc;
  for (int i = 0; i < 100; i++) {
    uint64_t key = 2 * i;
    std::ostringstream oss;
    oss << "Hello World. I am " << key;
    rc = btree.DoInsert(key, oss.str());
    EXPECT_TRUE(rc.IsOk());
  }
  for (int i = 0; i < 100; i++) {
    uint64_t key = 2 * i + 1;
    std::ostringstream oss;
    oss << "Hello World. I am " << key;
    rc = btree.DoInsert(key, oss.str());
    EXPECT_TRUE(rc.IsOk());
  }
  EXPECT_EQ(btree.size(), 200);

  // Test iterator
  {
    int cnt = 0;
    auto it = btree.begin();
    uint64_t prev = it.key();
    ++it;
    ++cnt;
    while (it != btree.end()) {
      uint64_t cur = it.key();
      std::string val = "Hello World. I am " + std::to_string(cur);
      EXPECT_TRUE(prev < cur);
      EXPECT_EQ(it.value(), val);
      prev = cur;
      ++it;
      ++cnt;
    }
    EXPECT_EQ(cnt, 200);
    // Now go backward
    for (int i = 0; i < 10; i++) {
      --it;
      EXPECT_EQ(199 - i, it.key());
    }
  }

  // Test search
  {
    MS_LOG(INFO) << "Locate key " << 100 << " Expect found.";
    auto r = btree.Search(100);
    auto &it = r.first;
    EXPECT_TRUE(r.second);
    EXPECT_EQ(it.key(), 100);
    EXPECT_EQ(it.value(), "Hello World. I am 100");
    MS_LOG(INFO) << "Locate key " << 300 << " Expect not found.";
    auto q = btree.Search(300);
    EXPECT_FALSE(q.second);
  }

  // Test duplicate key
  {
    rc = btree.DoInsert(100, "Expect error");
    EXPECT_EQ(rc, Status(StatusCode::kMDDuplicateKey));
  }
}

// Test concurrent insert.
TEST_F(MindDataTestBPlusTree, Test2) {
  Allocator<std::string> alloc(std::make_shared<SystemPool>());
  BPlusTree<uint64_t, std::string, Allocator<std::string>, std::less<>, mytraits> btree(alloc);
  TaskGroup vg;
  auto f = [&](int k) -> Status {
    TaskManager::FindMe()->Post();
    for (int i = 0; i < 100; i++) {
      uint64_t key = k * 100 + i;
      std::ostringstream oss;
      oss << "Hello World. I am " << key;
      Status rc = btree.DoInsert(key, oss.str());
      EXPECT_TRUE(rc.IsOk());
    }
    return Status::OK();
  };
  auto g = [&](int k) -> Status {
    TaskManager::FindMe()->Post();
    for (int i = 0; i < 1000; i++) {
      uint64_t key = rand() % 10000;
      ;
      auto it = btree.Search(key);
    }
    return Status::OK();
  };
  // Spawn multiple threads to do insert.
  for (int k = 0; k < 100; k++) {
    vg.CreateAsyncTask("Concurrent Insert", std::bind(f, k));
  }
  // Spawn a few threads to do random search.
  for (int k = 0; k < 2; k++) {
    vg.CreateAsyncTask("Concurrent search", std::bind(g, k));
  }
  vg.join_all();
  EXPECT_EQ(btree.size(), 10000);

  // Test iterator
  {
    int cnt = 0;
    auto it = btree.begin();
    uint64_t prev = it.key();
    ++it;
    ++cnt;
    while (it != btree.end()) {
      uint64_t cur = it.key();
      std::string val = "Hello World. I am " + std::to_string(cur);
      EXPECT_TRUE(prev < cur);
      EXPECT_EQ(it.value(), val);
      prev = cur;
      ++it;
      ++cnt;
    }
    EXPECT_EQ(cnt, 10000);
  }

  // Test search
  {
    MS_LOG(INFO) << "Locating key from 0 to 9999. Expect found.";
    for (int i = 0; i < 10000; i++) {
      auto r = btree.Search(i);
      EXPECT_TRUE(r.second);
      if (r.second) {
        auto &it = r.first;
        EXPECT_EQ(it.key(), i);
        std::string val = "Hello World. I am " + std::to_string(i);
        EXPECT_EQ(it.value(), val);
      }
    }
    MS_LOG(INFO) << "Locate key " << 10000 << ". Expect not found";
    auto q = btree.Search(10000);
    EXPECT_FALSE(q.second);
  }
}

TEST_F(MindDataTestBPlusTree, Test3) {
  Allocator<std::string> alloc(std::make_shared<SystemPool>());
  AutoIndexObj<std::string, Allocator<std::string>> ai(alloc);
  Status rc;
  rc = ai.insert("Hello World");
  EXPECT_TRUE(rc.IsOk());
  rc = ai.insert({"a", "b", "c"});
  EXPECT_TRUE(rc.IsOk());
  uint64_t min = ai.min_key();
  uint64_t max = ai.max_key();
  EXPECT_EQ(min, 0);
  EXPECT_EQ(max, 3);
  auto r = ai.Search(2);
  auto &it = r.first;
  EXPECT_EQ(it.value(), "b");
  MS_LOG(INFO) << "Dump all the values using [] operator.";
  for (uint64_t i = min; i <= max; i++) {
    MS_LOG(DEBUG) << ai[i] << std::endl;
  }
}

TEST_F(MindDataTestBPlusTree, Test4) {
  Allocator<int64_t> alloc(std::make_shared<SystemPool>());
  AutoIndexObj<int64_t, Allocator<int64_t>> ai(alloc);
  Status rc;
  for (int i = 0; i < 1000; i++) {
    rc = ai.insert(std::make_unique<int64_t>(i));
    EXPECT_TRUE(rc.IsOk());
  }
  // Test iterator
  {
    int cnt = 0;
    auto it = ai.begin();
    uint64_t prev = it.key();
    ++it;
    ++cnt;
    while (it != ai.end()) {
      uint64_t cur = it.key();
      EXPECT_TRUE(prev < cur);
      EXPECT_EQ(it.value(), cnt);
      prev = cur;
      ++it;
      ++cnt;
    }
    EXPECT_EQ(cnt, 1000);
  }
}

TEST_F(MindDataTestBPlusTree, TestPerfNoLocking) {
  AutoIndexObj<int64_t> btree;
  // No locking test
  btree.SetLocking(false);
  // Insert a million entries using the default traits.
  for (auto i = 0; i < 1000000; ++i) {
    ASSERT_TRUE(btree.insert(i));
  }
  std::cout << "Tree height : " << btree.GetHeight() << std::endl;
  std::cout << "Tree Order : " << btree.GetOrder() << std::endl;
  std::cout << "Number of leaves : " << btree.GetNumLeaves() << std::endl;
  std::cout << "Number of inner nodes : " << btree.GetNumInnerNodes() << std::endl;

  auto r = btree.Search(3);
  EXPECT_TRUE(r.second);
  r = btree.Search(999999);
  EXPECT_TRUE(r.second);
}
