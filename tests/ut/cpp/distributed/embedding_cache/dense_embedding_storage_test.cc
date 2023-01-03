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
#include <string>

#include "common/common_test.h"
#include "distributed/embedding_cache/embedding_storage/dense_embedding_storage.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace distributed {
namespace storage {
class TestDenseEmbeddingStorage : public UT::Common {
 public:
  TestDenseEmbeddingStorage() = default;
  virtual ~TestDenseEmbeddingStorage() = default;

  void SetUp() override {}
  void TearDown() override {}
};

using device::DeviceAddressPtr;
using device::cpu::CPUDeviceAddress;
/// Feature: test dense embedding storage all api.
/// Description: test dense embedding storage data structure and interface.
/// Expectation: all interface work normally or throw expectant exception.
TEST_F(TestDenseEmbeddingStorage, test_dense_embedding_storage) {
  int32_t embedding_key = 0;
  size_t embedding_dim = 8;
  size_t capacity = 100;
  DenseEmbeddingStorage<int, float, std::allocator<uint8_t>> embed_storage(embedding_key, embedding_dim, capacity);
  std::unique_ptr<float[]> embedding_table = std::make_unique<float[]>(capacity * embedding_dim);

  DeviceAddressPtr device_address =
    std::make_shared<CPUDeviceAddress>(embedding_table.get(), capacity * embedding_dim * sizeof(float));
  EXPECT_NE(device_address, nullptr);
  EXPECT_NO_THROW(embed_storage.Initialize(device_address.get()));

  size_t key_num = 10;
  std::vector<int> keys(key_num);
  std::iota(keys.begin(), keys.end(), 0);
  std::vector<float> embeddings_to_put(key_num * embedding_dim);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < embedding_dim; j++) {
      embeddings_to_put[i * embedding_dim + j] = static_cast<float>(i);
    }
  }

  EXPECT_EQ(embed_storage.Put(keys.data(), key_num, embeddings_to_put.data()), true);

  std::vector<float> embeddings_to_get(key_num * embedding_dim);
  EXPECT_EQ(embed_storage.Get(keys.data(), key_num, embeddings_to_get.data()), true);

  EXPECT_EQ(embeddings_to_get, embeddings_to_put);

  EXPECT_NO_THROW(embed_storage.Finalize());
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
