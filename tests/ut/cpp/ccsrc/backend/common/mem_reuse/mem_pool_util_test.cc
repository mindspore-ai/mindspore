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
#include <vector>

#include "common/common_test.h"
#include "include/runtime/memory/mem_pool/abstract_dynamic_mem_pool.h"

namespace mindspore {
namespace device {

class TestObjectPool : public UT::Common {
 public:
  TestObjectPool() = default;
  virtual ~TestObjectPool() = default;
};

/// Feature: test basic operation for object pool.
/// Description: test basic allocation.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestObjectPool, test_basic_operation) {
  ObjectPool<int> pool;
  auto obj1 = pool.Borrow();
  EXPECT_NE(obj1, nullptr);
  auto obj2 = pool.Borrow();
  EXPECT_NE(obj2, nullptr);
  pool.Return(obj1);
  pool.Return(obj2);
  auto obj3 = pool.Borrow();
  EXPECT_NE(obj3, nullptr);
  EXPECT_EQ(obj3, obj2);
  auto obj4 = pool.Borrow();
  EXPECT_NE(obj4, nullptr);
  EXPECT_EQ(obj4, obj1);
}
} // namespace device
} // namespace mindspore
