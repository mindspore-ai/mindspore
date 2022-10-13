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

#include "include/common/random.h"
#include "distributed/embedding_cache/embedding_store.h"

namespace mindspore {
namespace distributed {
namespace persistent {
class TestEmbeddingStore : public UT::Common {
 public:
  TestEmbeddingStore() = default;
  virtual ~TestEmbeddingStore() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test embedding store.
/// Description: test embedding store data structure and interface.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestEmbeddingStore, test_embedding_store_simple_case) {
  size_t emb_dim = 3;
  size_t vocab_size = 3;
  size_t vocab_cache_size = 1;
  size_t shape_size = vocab_size * emb_dim;
  size_t cache_shape_size = vocab_cache_size * emb_dim;
  std::string name = "fake";

  auto emb_store = std::make_shared<distributed::EmbeddingStore<int32_t, float>>(name, vocab_cache_size, emb_dim);
  EXPECT_NO_THROW(emb_store->Initialize());

  std::vector<float> input(cache_shape_size);
  std::vector<float> values(shape_size);
  size_t key_num = 3;
  std::vector<int32_t> keys{0, 1, 2};

  // Get keys not exists
  EXPECT_FALSE(emb_store->Get(input.data(), key_num, keys.data(), values.data()));

  // Put key&value out of cache range
  for (int i = 0; i < shape_size; i++) {
    values[i] = 1.0 * i;
  }
  EXPECT_TRUE(emb_store->Put(input.data(), key_num, keys.data(), values.data()));
  for (int i = 2 * emb_dim; i < shape_size; i++) {
    EXPECT_FLOAT_EQ(1.0 * i, input[i - 2 * emb_dim]);
  }

  // Get all key&value
  for (int i = 0; i < shape_size; i++) {
    values[i] = 0;
  }
  EXPECT_TRUE(emb_store->Get(input.data(), key_num, keys.data(), values.data()));
  for (int i = 0; i < shape_size; i++) {
    EXPECT_FLOAT_EQ(1.0 * i, values[i]);
  }
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore
