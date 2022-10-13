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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORE_H_

#include <map>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "distributed/persistent/storage/storage.h"
#include "distributed/embedding_cache/embedding_lru_cache.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
// Default path to save embedding table.
static constexpr auto kDefaultEmbeddingRemoteStoragePath = "./embedding_storage";

static constexpr auto kEnvEmbeddingRemoteStoragePath = "MS_EMBEDDING_REMOTE_STORAGE_PATH";

std::string GetEmbeddingRemoteStoragePath();

// This class uses external storage to store Parameter of embedding.
template <typename K, typename V>
class BACKEND_EXPORT EmbeddingStore {
 public:
  EmbeddingStore(std::string name, size_t cache_capacity, size_t emb_dim)
      : name_(name), cache_capacity_(cache_capacity), emb_dim_(emb_dim) {}
  ~EmbeddingStore() = default;

  bool Initialize();
  bool Finalize() { return true; }

  // Get values which is indexed by keys at input. Input is a tensor data address from Parameter of embedding.
  // Values save the get result. Keys is lookup indices.
  // When keys not exist in input, will get values from persistent storage.
  bool Get(const void *input, size_t key_num, const void *keys, void *values);

  // Get values which is indexed by keys at persistent storage.
  bool Get(size_t key_num, const void *keys, void *values);

  // Put values which is indexed by keys to input. Input is a tensor data address from Parameter of embedding.
  // Values is data to be update to input. Keys is update indices.
  // When input is full, save evicted values to persistent storage.
  bool Put(void *input, size_t key_num, const void *keys, const void *values);

  // Flush input to persistent storage.
  bool Flush(void *input);

 private:
  // A unique name for this embedding store.
  std::string name_;

  // Cache the keys of Parameter of embedding.
  std::unique_ptr<EmbeddingCache> cache_;

  // A persistent storage to provide external storage for cache.
  std::unique_ptr<storage::StorageBase> storage_;

  size_t cache_capacity_;

  // The count of element of value.
  size_t emb_dim_;

  // Total size of bytes of value.
  size_t value_size_;

  // Total size of bytes of key.
  size_t key_size_;

  // Vector to save miss keys when get from cache.
  std::vector<K> cache_miss_keys_;
  // Vector to save miss indices when get from cache.
  std::vector<size_t> cache_miss_indices_;
  // Vector to save miss indices when get from storage.
  std::vector<size_t> storage_miss_indices_;
  // Buffer to save data read from storage.
  std::vector<uint8_t> storage_output_buf_;
  // Buffer to save evicted keys when put from cache.
  std::vector<K> evicted_keys_;
  // Buffer to save evicted values when put from cache.
  std::vector<uint8_t> evicted_values_buf_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORE_H_
