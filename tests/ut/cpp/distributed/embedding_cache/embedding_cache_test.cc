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

  // Since EmbeddingCacheTableManager is singleton, multiple unit test will fail as interacting with each other.
  // Feature: test warm up host cache async.
  // Description: test warm up process.
  // Expectation: all interface work normally and can not throw exception.
  embedding_size = 100;
  param_key = 1;

  // Clean table.
  auto &hash_tables = embedding_cache_manager.hash_tables();
  hash_tables.erase(param_name);
  hash_tables.erase(new_param_name);
  hash_tables.erase(accu_param_name);
  EXPECT_EQ(0, hash_tables.size());

  // Prepare table.
  EXPECT_NO_THROW(
    embedding_cache_manager.InsertHashTableSize(param_name, vocab_cache_size, embedding_size, vocab_size, param_key));
  EXPECT_EQ(1, hash_tables.size());
  // Find host table, and malloc memory for host_address.
  const auto &iter = std::find_if(hash_tables.begin(), hash_tables.end(),
                                  [this, param_key](const auto &data) { return data.second.param_key_ == param_key; });
  auto host_table_info_ptr = &(iter->second);
  auto &host_address = host_table_info_ptr->host_address;
  int host_cache_size = 1000;
  // Embedding host cache not use first position.
  size_t host_length = (host_cache_size + 2) * embedding_size;
  auto host_hash_table_addr = std::make_unique<float[]>(host_length);
  host_address = host_hash_table_addr.get();
  auto ret = memset_s(host_address, host_length * sizeof(float), 0, host_length * sizeof(float));
  EXPECT_EQ(ret, EOK);

  auto host_hash_map = std::make_shared<EmbeddingHashMap>(host_cache_size + 2);
  embedding_cache_manager.set_host_hash_map(host_hash_map);

  // Prepare tensors.
  const int key_size = host_cache_size;
  std::vector<int> key_vec(key_size);
  std::iota(key_vec.begin(), key_vec.end(), 0);
  auto key_tensor_ptr = std::make_shared<tensor::Tensor>(key_vec);
  int value_size = embedding_size;
  int value_shape_size = key_size * value_size;
  std::vector<int> value_vec(value_shape_size);
  std::iota(value_vec.begin(), value_vec.end(), 1);
  auto value_tensor_ptr = std::make_shared<tensor::Tensor>(
    TypeId::kNumberTypeUInt32, std::vector<int64_t>({key_size, value_size}), value_vec.data(), value_shape_size << 2);
  embedding_cache_manager.StoreWarmUpPtr(param_key, key_tensor_ptr, value_tensor_ptr, key_tensor_ptr);
  auto host_cache_ptrs = embedding_cache_manager.host_cache_ptrs();
  EXPECT_EQ(1, host_cache_ptrs.size());
  int *host_address_ptr = reinterpret_cast<int *>(host_address);
  for (int i = 0; i != host_length; i++) {
    EXPECT_EQ(0, *(host_address_ptr + i));
  }

  // Start warm up process.
  bool status = embedding_cache_manager.WaitForWarmUpHostCacheComplete();

  // Assert status and caches, note first position is 0 and last position is not in warm up range.
  EXPECT_EQ(true, status);
  for (int i = 0; i != value_size; i++) {
    EXPECT_EQ(0, *(host_address_ptr + i));
  }
  for (int i = value_size, end = host_length - value_size; i != end; i++) {
    EXPECT_EQ(i - value_size + 1, *(host_address_ptr + i));
  }
  for (int i = host_length - value_size; i != host_length; i++) {
    EXPECT_EQ(0, *(host_address_ptr + i));
  }
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore
