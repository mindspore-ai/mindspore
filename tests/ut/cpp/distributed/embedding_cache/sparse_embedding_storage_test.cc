/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <string>

#include "common/common_test.h"
#include "distributed/embedding_cache/embedding_storage/sparse_embedding_storage.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/cpu_hash_table.h"

namespace mindspore {
namespace distributed {
namespace storage {
class TestSparseEmbeddingStorage : public UT::Common {
 public:
  TestSparseEmbeddingStorage() = default;
  virtual ~TestSparseEmbeddingStorage() = default;

  void SetUp() override {}
  void TearDown() override {}
};

using device::DeviceAddressPtr;
using device::cpu::CPUDeviceAddress;
using device::cpu::CPUHashTable;
/// Feature: test sparse embedding storage all api.
/// Description: test sparse embedding storage data structure and interface.
/// Expectation: all interface work normally or throw expectant exception.
TEST_F(TestSparseEmbeddingStorage, DISABLED_test_sparse_embedding_storage) {
  int32_t embedding_key = 0;
  size_t embedding_dim = 8;
  size_t capacity = 15;
  SparseEmbeddingStorage<int, float, std::allocator<uint8_t>> embed_storage(embedding_key, embedding_dim, capacity);
  std::unique_ptr<float[]> embedding_table = std::make_unique<float[]>(capacity * embedding_dim);

  DeviceAddressPtr device_address =
    std::make_shared<CPUDeviceAddress>(embedding_table.get(), capacity * embedding_dim * sizeof(float));
  EXPECT_NE(device_address, nullptr);
  UserDataPtr user_data = std::make_shared<UserData>();
  user_data->set<CPUHashTable<int, float>>(kUserDataData, std::make_shared<CPUHashTable<int, float>>(embedding_dim));
  device_address->set_user_data(user_data);

  EXPECT_NO_THROW(embed_storage.Initialize(device_address.get()));

  size_t key_num = 10;
  std::vector<float> embeddings_to_get(key_num * embedding_dim);
  std::vector<int> keys1(key_num);
  std::iota(keys1.begin(), keys1.end(), 0);
  std::vector<float> embeddings_to_put1(key_num * embedding_dim);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < embedding_dim; j++) {
      embeddings_to_put1[i * embedding_dim + j] = static_cast<float>(i);
    }
  }

  // First put and get.
  EXPECT_EQ(embed_storage.Put({keys1.data(), key_num * sizeof(int)},
                              {embeddings_to_put1.data(), embeddings_to_put1.size() * sizeof(float)}),
            true);
  EXPECT_EQ(embed_storage.Get({keys1.data(), key_num * sizeof(int)},
                              {embeddings_to_get.data(), embeddings_to_get.size() * sizeof(float)}),
            true);
  EXPECT_EQ(embeddings_to_get, embeddings_to_put1);

  // Second put and get, cache will update and interact with persistent storage.
  std::vector<int> keys2(key_num);
  std::iota(keys2.begin(), keys2.end(), keys2.size());
  std::vector<float> embeddings_to_put2(key_num * embedding_dim);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < embedding_dim; j++) {
      embeddings_to_put2[i * embedding_dim + j] = static_cast<float>(i + key_num);
    }
  }

  EXPECT_EQ(embed_storage.Put({keys2.data(), key_num * sizeof(int)},
                              {embeddings_to_put2.data(), embeddings_to_put2.size() * sizeof(float)}),
            true);
  EXPECT_EQ(embed_storage.Get({keys2.data(), key_num * sizeof(int)},
                              {embeddings_to_get.data(), embeddings_to_get.size() * sizeof(float)}),
            true);
  EXPECT_EQ(embeddings_to_get, embeddings_to_put2);

  // Get the value first put into embedding storage and interact with persistent storage.
  EXPECT_EQ(embed_storage.Get({keys1.data(), key_num * sizeof(int)},
                              {embeddings_to_get.data(), embeddings_to_get.size() * sizeof(float)}),
            true);
  EXPECT_EQ(embeddings_to_get, embeddings_to_put1);

  EXPECT_NO_THROW(embed_storage.Finalize());
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
