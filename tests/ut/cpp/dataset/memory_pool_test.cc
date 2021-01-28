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

#include "minddata/dataset/util/memory_pool.h"
#include "minddata/dataset/util/circular_pool.h"
#include "minddata/dataset/util/system_pool.h"
#include "minddata/dataset/util/allocator.h"
#include "common/common.h"
#include "gtest/gtest.h"

using namespace mindspore::dataset;

class MindDataTestMemoryPool : public UT::Common {
 public:
  std::shared_ptr<MemoryPool> mp_;
  MindDataTestMemoryPool() {}

  void SetUp() {
    Status rc = CircularPool::CreateCircularPool(&mp_, 1, 1, true);
    ASSERT_TRUE(rc.IsOk());
  }
};

TEST_F(MindDataTestMemoryPool, DumpPoolInfo) {
  MS_LOG(DEBUG) << *(std::dynamic_pointer_cast<CircularPool>(mp_)) << std::endl;
}

TEST_F(MindDataTestMemoryPool, TestOperator1) {
  Status rc;
  int *p = new (&rc, mp_) int;
  ASSERT_TRUE(rc.IsOk());
  *p = 2048;
  ::operator delete(p, mp_);
}

TEST_F(MindDataTestMemoryPool, TestOperator3) {
  Status rc;
  int *p = new (&rc, mp_) int[100];
  ASSERT_TRUE(rc.IsOk());
  for (int i = 0; i < 100; i++) {
    p[i] = i;
  }
  for (int i = 0; i < 100; i++) {
    ASSERT_EQ(p[i], i);
  }
}

TEST_F(MindDataTestMemoryPool, TestAllocator) {
  class A {
   public:
    explicit A(int x) : a(x) {}
    int val_a() const { return a; }

   private:
    int a;
  };
  Allocator<A> alloc(mp_);
  std::shared_ptr<A> obj_a = std::allocate_shared<A>(alloc, 3);
  int v = obj_a->val_a();
  ASSERT_EQ(v, 3);
  MS_LOG(DEBUG) << *(std::dynamic_pointer_cast<CircularPool>(mp_)) << std::endl;
}

TEST_F(MindDataTestMemoryPool, TestMemGuard) {
  MemGuard<uint8_t> mem;
  // Try some large value.
  int64_t sz = 5LL * 1024LL * 1024LL * 1024LL;
  Status rc = mem.allocate(sz);
  ASSERT_TRUE(rc.IsOk() || rc == StatusCode::kMDOutOfMemory);
  if (rc.IsOk()) {
    // Try write a character half way.
    auto *p = mem.GetMutablePointer();
    p[sz / 2] = 'a';
  }
}
