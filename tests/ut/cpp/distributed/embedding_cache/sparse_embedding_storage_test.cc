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
using ExportData = std::vector<std::shared_ptr<std::vector<char>>>;
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
  user_data->set<CPUHashTable<int, float>>(kUserDataData,
                                           std::make_shared<CPUHashTable<int, float>>(embedding_dim, 0.0));
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

/// Feature: sparse embedding storage supports export slice api.
/// Description: test sparse embedding storage data export slice api.
/// Expectation: all interface work normally or throw expectant exception.
TEST_F(TestSparseEmbeddingStorage, DISABLED_test_sparse_embedding_storage_export_slice) {
  int32_t embedding_key = 0;
  size_t embedding_dim = 1024;
  size_t capacity = 256 * 2;

  size_t key_num = capacity * 2;
  size_t slice_size_in_mb = 1;
  size_t slice_num = (slice_size_in_mb << 20) / (embedding_dim * sizeof(float));

  SparseEmbeddingStorage<int, float, std::allocator<uint8_t>> embed_storage(embedding_key, embedding_dim, capacity);
  std::unique_ptr<float[]> embedding_table = std::make_unique<float[]>(capacity * embedding_dim);

  DeviceAddressPtr device_address =
    std::make_shared<CPUDeviceAddress>(embedding_table.get(), capacity * embedding_dim * sizeof(float));
  EXPECT_NE(device_address, nullptr);
  UserDataPtr user_data = std::make_shared<UserData>();
  user_data->set<CPUHashTable<int, float>>(kUserDataData,
                                           std::make_shared<CPUHashTable<int, float>>(embedding_dim, 0.0));
  device_address->set_user_data(user_data);

  EXPECT_NO_THROW(embed_storage.Initialize(device_address.get()));

  std::vector<int> keys1(key_num / 2);
  std::iota(keys1.begin(), keys1.end(), 0);
  std::vector<float> embeddings_to_put1(key_num / 2 * embedding_dim);

  for (size_t i = 0; i < key_num / 2; i++) {
    for (size_t j = 0; j < embedding_dim; j++) {
      embeddings_to_put1[i * embedding_dim + j] = static_cast<float>(i);
    }
  }

  std::vector<int> keys2(key_num / 2);
  std::iota(keys2.begin(), keys2.end(), capacity);
  std::vector<float> embeddings_to_put2(key_num / 2 * embedding_dim);

  for (size_t i = 0; i < key_num / 2; i++) {
    for (size_t j = 0; j < embedding_dim; j++) {
      embeddings_to_put2[i * embedding_dim + j] = static_cast<float>(i + capacity);
    }
  }

  std::map<int, std::vector<float>> keys_values;
  for (size_t i = 0; i < key_num / 2; ++i) {
    keys_values.emplace(keys1[i], std::vector<float>(embeddings_to_put1.begin() + i * embedding_dim,
                                                     embeddings_to_put1.begin() + (i + 1) * embedding_dim));

    keys_values.emplace(keys2[i], std::vector<float>(embeddings_to_put2.begin() + i * embedding_dim,
                                                     embeddings_to_put2.begin() + (i + 1) * embedding_dim));
  }
  EXPECT_EQ(keys_values.size(), key_num);

  EXPECT_EQ(embed_storage.Put({keys1.data(), key_num / 2 * sizeof(int)},
                              {embeddings_to_put1.data(), embeddings_to_put1.size() * sizeof(float)}),
            true);
  EXPECT_EQ(embed_storage.Put({keys2.data(), key_num / 2 * sizeof(int)},
                              {embeddings_to_put2.data(), embeddings_to_put2.size() * sizeof(float)}),
            true);

  std::vector<ExportData> export_data;
  bool last_slice = false;
  while (!last_slice) {
    ExportData ret;
    EXPECT_NO_THROW(ret = embed_storage.ExportSlice(false, &last_slice, slice_size_in_mb));
    EXPECT_EQ(ret[0]->size(), sizeof(int) * slice_num);
    EXPECT_EQ(ret[1]->size(), sizeof(float) * embedding_dim * slice_num);
    EXPECT_EQ(ret[2]->size(), sizeof(HashTableElementStatus) * slice_num);
    export_data.push_back(ret);
  }
  EXPECT_EQ(export_data.size(), key_num / slice_num);

  std::map<int, std::vector<float>> keys_values_to_check;
  for (auto &item : export_data) {
    auto &key_data = item[0];
    auto &value_data = item[1];
    size_t slice_key_num = key_data->size() / sizeof(int);
    int *key_ptr = reinterpret_cast<int *>(key_data->data());
    float *value_ptr = reinterpret_cast<float *>(value_data->data());

    for (size_t i = 0; i < slice_key_num; i++) {
      keys_values_to_check.emplace(
        key_ptr[i], std::vector<float>(value_ptr + i * embedding_dim, value_ptr + (i + 1) * embedding_dim));
    }
  }

  EXPECT_EQ(keys_values_to_check.size(), key_num);
  EXPECT_EQ(keys_values_to_check, keys_values);

  EXPECT_NO_THROW(embed_storage.Finalize());
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
