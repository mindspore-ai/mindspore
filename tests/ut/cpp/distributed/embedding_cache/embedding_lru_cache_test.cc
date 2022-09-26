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

#include "common/common_test.h"

#include <memory>
#include <map>
#include <vector>
#include <string>
#include <random>

#include "distributed/embedding_cache/embedding_lru_cache.h"

namespace mindspore {
namespace distributed {
namespace persistent {
class TestEmbeddingLRUCache : public UT::Common {
 public:
  TestEmbeddingLRUCache() = default;
  virtual ~TestEmbeddingLRUCache() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test lru cache.
/// Description: test embedding lru cache data structure and interface.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestEmbeddingLRUCache, test_lru_cache_base_api) {
  distributed::LRUCache<int, int> cache(1);
  cache.Put(8, 987);
  EXPECT_TRUE(cache.Exists(8));
  EXPECT_EQ(987, cache.Get(8));
  EXPECT_EQ(1, cache.Size());
  EXPECT_TRUE(cache.IsFull());
}

/// Feature: test lru cache.
/// Description: test embedding lru cache data structure and interface of simple use case.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestEmbeddingLRUCache, test_lru_cache_simple_case) {
  int all_num = 100;
  int capacity = 50;

  distributed::LRUCache<int, int> cache(capacity);

  for (int i = 0; i < all_num; ++i) {
    cache.Put(i, i);
  }

  for (int i = 0; i < all_num - capacity; ++i) {
    EXPECT_FALSE(cache.Exists(i));
  }

  for (int i = all_num - capacity; i < all_num; ++i) {
    EXPECT_TRUE(cache.Exists(i));
    EXPECT_EQ(i, cache.Get(i));
  }

  size_t size = cache.Size();
  EXPECT_EQ(capacity, size);
}

/// Feature: test lru cache.
/// Description: test embedding lru cache data structure and interface when miss key.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestEmbeddingLRUCache, test_lru_cache_miss_key) {
  distributed::LRUCache<int, int> cache(1);
  EXPECT_ANY_THROW(cache.Get(5));
}

/// Feature: test embedding lru cache.
/// Description: test embedding lru cache data structure and interface of simple use case.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestEmbeddingLRUCache, test_emb_lru_cache_simple_case) {
  size_t emb_dim = 256;
  size_t vocab_size = 1;
  size_t value_size = emb_dim * sizeof(float);
  size_t shape_size = vocab_size * emb_dim;
  size_t cache_capacity = 1;

  auto cache = std::make_unique<EmbeddingLRUCache<int32_t, float>>(cache_capacity, value_size);
  EXPECT_NO_THROW(cache->Initialize());

  std::vector<float> input;
  for (int i = 0; i < shape_size; i++) {
    input.emplace_back(1.0 * i);
  }
  size_t key_num = 1;
  std::vector<int32_t> keys{0};
  std::vector<float> values(shape_size);
  size_t miss_num = 0;
  std::vector<int32_t> miss_keys(1);
  std::vector<size_t> miss_indices(1);

  // Get not exists key
  EXPECT_TRUE(
    cache->Get(input.data(), key_num, keys.data(), values.data(), &miss_num, miss_keys.data(), miss_indices.data()));
  EXPECT_EQ(1, miss_num);
  EXPECT_EQ(0, miss_keys[0]);
  EXPECT_EQ(1, miss_indices.size());
  EXPECT_EQ(0, miss_indices[0]);

  // Put key&value to input
  for (int i = 0; i < shape_size; i++) {
    values[i] = 3.0 * i;
  }
  size_t evicted_num = 0;
  std::vector<int32_t> evicted_keys(1);
  std::vector<float> evicted_values(shape_size);
  EXPECT_TRUE(cache->Put(input.data(), key_num, keys.data(), values.data(), &evicted_num, evicted_keys.data(),
                         evicted_values.data()));
  EXPECT_EQ(0, evicted_num);
  for (int i = 0; i < shape_size; i++) {
    EXPECT_FLOAT_EQ(3.0 * i, input[i]);
  }

  // Put new key&value to input
  keys[0] = 42;
  for (int i = 0; i < shape_size; i++) {
    values[i] = 5.0 * i;
  }
  EXPECT_TRUE(cache->Put(input.data(), key_num, keys.data(), values.data(), &evicted_num, evicted_keys.data(),
                         evicted_values.data()));
  EXPECT_EQ(1, evicted_num);
  EXPECT_EQ(1, evicted_keys.size());
  for (int i = 0; i < shape_size; i++) {
    EXPECT_FLOAT_EQ(5.0 * i, input[i]);
  }
  for (int i = 0; i < shape_size; i++) {
    EXPECT_FLOAT_EQ(3.0 * i, evicted_values[i]);
  }

  // Get old key will miss
  keys[0] = 0;
  EXPECT_TRUE(
    cache->Get(input.data(), key_num, keys.data(), values.data(), &miss_num, miss_keys.data(), miss_indices.data()));
  EXPECT_EQ(1, miss_num);
  EXPECT_EQ(0, miss_keys[0]);
  EXPECT_EQ(1, miss_indices.size());
  EXPECT_EQ(0, miss_indices[0]);
  // value not change after get
  for (int i = 0; i < shape_size; i++) {
    EXPECT_FLOAT_EQ(5.0 * i, input[i]);
  }
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore
