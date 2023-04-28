/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <list>

#include "common/common_test.h"
#include "distributed/embedding_cache/cache_strategy/lru_cache.h"

namespace mindspore {
namespace distributed {
class TestLRUCache : public UT::Common {
 public:
  TestLRUCache() = default;
  virtual ~TestLRUCache() = default;

  void SetUp() override {}
  void TearDown() override {}
};

using Element = typename LRUCache<int, int>::Element;
/// Feature: test lru cache all api.
/// Description: test lru cache data structure and interface.
/// Expectation: all interface work normally or throw expectant exception.
TEST_F(TestLRUCache, test_lru_cache) {
  distributed::LRUCache<int, int> cache(5);
  EXPECT_EQ(cache.capacity(), 5);
  std::vector<Element> origin_elements = {{1, 11}, {2, 22}, {3, 33}};
  for (const auto &item : origin_elements) {
    EXPECT_NO_THROW(cache.Put(item.first, item.second));
  }
  EXPECT_TRUE(cache.Exists(1));
  EXPECT_TRUE(cache.Exists(2));
  EXPECT_TRUE(cache.Exists(3));
  EXPECT_FALSE(cache.Exists(4));

  EXPECT_EQ(origin_elements.size(), cache.size());
  EXPECT_FALSE(cache.IsFull());

  std::list<Element> cache_elements = cache.Export();
  auto vec_reverse_iter = origin_elements.rbegin();
  for (auto list_iter = cache_elements.begin(); list_iter != cache_elements.end(); ++list_iter, ++vec_reverse_iter) {
    EXPECT_EQ((*list_iter), (*vec_reverse_iter));
  }

  int value = 0;
  EXPECT_EQ(cache.Get(1, &value), true);
  EXPECT_EQ(value, 11);
  EXPECT_EQ((cache.Export().front()), (std::pair<int, int>(1, 11)));

  EXPECT_EQ(cache.Get(2, &value), true);
  EXPECT_EQ(value, 22);
  EXPECT_EQ((cache.Export().front()), (std::pair<int, int>(2, 22)));

  EXPECT_EQ(cache.Get(3, &value), true);
  EXPECT_EQ(value, 33);
  EXPECT_EQ((cache.Export().front()), (std::pair<int, int>(3, 33)));

  std::vector<Element> evict_elements;
  EXPECT_NO_THROW(cache.TryEvict(3, &evict_elements));
  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(evict_elements.size(), 1);
  EXPECT_EQ((evict_elements.front()), (std::pair<int, int>(1, 11)));

  EXPECT_EQ((cache.Export().front()), (std::pair<int, int>(3, 33)));
  EXPECT_EQ(cache.Get(2, &value), true);
  EXPECT_EQ(value, 22);
  EXPECT_NO_THROW(cache.Put(1, 11));

  std::list<Element> new_cache_elements = cache.Export();
  auto vec_iter = origin_elements.begin();
  for (auto list_iter = new_cache_elements.begin(); list_iter != new_cache_elements.end(); ++list_iter, ++vec_iter) {
    EXPECT_EQ((*list_iter), (*vec_iter));
  }

  EXPECT_EQ(cache.Get(4, &value), false);
  EXPECT_THROW(cache.TryEvict(6, &evict_elements), std::runtime_error);
}
}  // namespace distributed
}  // namespace mindspore