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
#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"

namespace mindspore {
namespace distributed {
namespace persistent {
class TestEmbeddingCache : public UT::Common {
 public:
  TestEmbeddingCache() = default;
  virtual ~TestEmbeddingCache() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test embedding cache.
/// Description: test embedding cache data structure and interface.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestEmbeddingCache, test_embedding_cache) {
  auto &embedding_cache_manager = distributed::EmbeddingCacheTableManager::GetInstance();
  std::string param_name = "network.embedding_table";
  std::string new_param_name = "network.deep_embedding_table";
  std::string accu_param_name = "network.moment.deep_embedding_table";
  size_t vocab_cache_size = 5000;
  size_t embedding_size = 16;
  size_t vocab_size = 10000;
  int32_t param_key = 0;
  int32_t accu_param_key = 1;

  EXPECT_NO_THROW(
    embedding_cache_manager.InsertHashTableSize(param_name, vocab_cache_size, embedding_size, vocab_size, param_key));

  EXPECT_NO_THROW(embedding_cache_manager.ReInsertHashTableSize(new_param_name, param_name));

  EXPECT_NO_THROW(embedding_cache_manager.CloneHashTable(accu_param_name, accu_param_key, new_param_name, param_key));

  EXPECT_EQ(true, embedding_cache_manager.IsEmbeddingCacheTable(accu_param_name));

  EXPECT_NO_THROW(embedding_cache_manager.set_batch_ids_num(16000));

  EXPECT_NO_THROW(embedding_cache_manager.cache_indices_lower_bound());
}

/// Feature: test the random number generator.
/// Description: test generate random numbers continuously.
/// Expectation: the specified random number are generated successfully.
TEST_F(TestEmbeddingCache, test_random_number_gen) {
  using T = float;
  using Generator = random::Philox;
  using Distribution = random::UniformDistribution<double>;

  const std::uint64_t seed = 0;
  const size_t skip = 0;
  std::unique_ptr<distributed::RandomGenerator<T, Generator, Distribution>> rnd_gen =
    std::make_unique<distributed::RandomGenerator<T, Generator, Distribution>>(seed, skip);

  const double distri_a = 0.0;
  const double distri_b = 1.0;
  rnd_gen->Initialize(distri_a, distri_b);

  const size_t count = 10;
  std::set<T> numbers;
  for (size_t i = 0; i < count; ++i) {
    auto num = rnd_gen->Next();
    auto rt = (num >= distri_a && num <= distri_b);
    ASSERT_TRUE(rt);

    numbers.insert(num);
  }
  ASSERT_TRUE(numbers.size() == count);
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore
