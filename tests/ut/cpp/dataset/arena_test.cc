/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <string>
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/arena.h"
#include "minddata/dataset/util/system_pool.h"
#include "common/common.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestArena : public UT::Common {
 public:
    MindDataTestArena() {}
};

/// Feature: Arena
/// Description: Test Arena basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestArena, Test1) {
  std::shared_ptr<Arena> mp;
  Status rc = Arena::CreateArena(&mp);
  ASSERT_TRUE(rc.IsOk());
  std::vector<void *> v;

  srand(time(NULL));
  for (int i = 0; i < 1000; i++) {
    uint64_t sz = rand() % 1048576;
    void *ptr = nullptr;
    ASSERT_TRUE(mp->Allocate(sz, &ptr));
    v.push_back(ptr);
  }
  for (int i = 0; i < 1000; i++) {
    mp->Deallocate(v.at(i));
  }
  MS_LOG(DEBUG) << *mp;
}

/// Feature: Arena
/// Description: Test copy and move
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestArena, Test2) {
  std::shared_ptr<Arena> arena;
  Status rc = Arena::CreateArena(&arena);
  std::shared_ptr<MemoryPool> mp = std::static_pointer_cast<MemoryPool>(arena);
  auto alloc = Allocator<int>(mp);
  ASSERT_TRUE(rc.IsOk());
  std::vector<int, Allocator<int>> v(alloc);
  v.reserve(1000);
  for (auto i = 0; i < 1000; ++i) {
    v.push_back(i);
  }
  // Test copy
  std::vector<int, Allocator<int>> w(v, SystemPool::GetAllocator<int>());
  auto val = w.at(10);
  EXPECT_EQ(val, 10);
  // Test move
  std::vector<int, Allocator<int>> s(std::move(v), SystemPool::GetAllocator<int>());
  val = s.at(100);
  EXPECT_EQ(val, 100);
  EXPECT_EQ(v.size(), 0);
}
