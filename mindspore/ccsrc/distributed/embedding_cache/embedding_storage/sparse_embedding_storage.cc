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

#include "distributed/embedding_cache/embedding_storage/sparse_embedding_storage.h"
#include <memory>
#include <vector>

namespace mindspore {
namespace distributed {
namespace storage {
template <typename KeyType, typename ValueType, typename Allocator>
void SparseEmbeddingStorage<KeyType, ValueType, Allocator>::Initialize(const DeviceAddress *device_address) {
  MS_EXCEPTION_IF_NULL(device_address);
  EmbeddingStorage<KeyType, ValueType, Allocator>::Initialize(device_address);

  auto user_data = device_address->user_data();
  MS_EXCEPTION_IF_NULL(user_data);
  hash_table_ = user_data->get<HashTable>(kUserDataData).get();
  MS_EXCEPTION_IF_NULL(hash_table_);
}

template <typename KeyType, typename ValueType, typename Allocator>
void SparseEmbeddingStorage<KeyType, ValueType, Allocator>::Finalize() {
  hash_table_ = nullptr;
  EmbeddingStorage<KeyType, ValueType, Allocator>::Finalize();
}

template <typename KeyType, typename ValueType, typename Allocator>
bool SparseEmbeddingStorage<KeyType, ValueType, Allocator>::Get(const ConstDataWithLen &keys,
                                                                const DataWithLen &values) {
  const KeyType *keys_data = reinterpret_cast<const KeyType *>(keys.data_);
  ValueType *values_data = reinterpret_cast<ValueType *>(values.data_);
  size_t key_num = keys.data_len_ / sizeof(KeyType);
  if (values.data_len_ < key_num * this->embedding_dim_ * sizeof(ValueType)) {
    MS_LOG(EXCEPTION) << "The value buffer length is insufficient.";
  }
  MS_EXCEPTION_IF_NULL(keys_data);
  MS_EXCEPTION_IF_NULL(values_data);
  MS_EXCEPTION_IF_NULL(hash_table_);

  // 1. Query cache to analyse the information of cache hit and miss keys, update the positions of cache hit elements in
  // the cache (cache refresh).
  size_t cache_miss_cnt = 0;
  size_t *cache_miss_offsets = this->template AllocateMemory<size_t>(sizeof(size_t) * key_num);
  MS_EXCEPTION_IF_NULL(cache_miss_offsets);
  bool *cache_hit = this->template AllocateMemory<bool>(sizeof(bool) * key_num);
  MS_EXCEPTION_IF_NULL(cache_hit);
  QueryCache(keys_data, key_num, cache_miss_offsets, &cache_miss_cnt, cache_hit);

  // 2. Copy the embeddings from cache to the returned values for cache hit keys.
  for (size_t i = 0; i < key_num; i++) {
    if (!cache_hit[i]) {
      continue;
    }
    RETURN_IF_FALSE_WITH_LOG(
      hash_table_->Find(keys_data + i, 1, false, values_data + this->embedding_dim_ * i, nullptr),
      "Find key from hash table failed.");
  }

  if (cache_miss_cnt == 0) {
    return true;
  }

  // 3. Reserve space for cache miss keys in the cache (if there is enough space in the cache, then do nothing), write
  // the evicted element to persistent storage, and record the space in the cache, using the space in the cache first.
  RETURN_IF_FALSE_WITH_LOG(TryEvict(cache_miss_cnt), "Reserve space for miss keys failed.");

  // 4. Insert the cache miss elements into the cache from persistent storage, and copy them to the returned values.
  RETURN_IF_FALSE_WITH_LOG(InsertMissCacheFromStorage(keys_data, cache_miss_offsets, cache_miss_cnt, values_data),
                           "Insert the cache miss elements into the cache from persistent storage failed.");

  this->FreeMemory(cache_hit);
  this->FreeMemory(cache_miss_offsets);
  return true;
}

template <typename KeyType, typename ValueType, typename Allocator>
bool SparseEmbeddingStorage<KeyType, ValueType, Allocator>::Put(const ConstDataWithLen &keys,
                                                                const ConstDataWithLen &values) {
  const KeyType *keys_data = reinterpret_cast<const KeyType *>(keys.data_);
  const ValueType *values_data = reinterpret_cast<const ValueType *>(values.data_);
  size_t key_num = keys.data_len_ / sizeof(KeyType);
  if (values.data_len_ != key_num * this->embedding_dim_ * sizeof(ValueType)) {
    MS_LOG(EXCEPTION) << "The value length is invalid, expected length["
                      << key_num * this->embedding_dim_ * sizeof(ValueType) << "], but got[" << values.data_len_ << "]";
  }
  MS_EXCEPTION_IF_NULL(keys_data);
  MS_EXCEPTION_IF_NULL(values_data);
  MS_EXCEPTION_IF_NULL(hash_table_);

  // 1. Query cache to analyse the information of cache hit and miss keys, update the positions of cache hit elements in
  // the cache (cache refresh).
  size_t cache_miss_cnt = 0;
  size_t *cache_miss_offsets = this->template AllocateMemory<size_t>(sizeof(size_t) * key_num);
  MS_EXCEPTION_IF_NULL(cache_miss_offsets);
  bool *cache_hit = this->template AllocateMemory<bool>(sizeof(bool) * key_num);
  MS_EXCEPTION_IF_NULL(cache_hit);
  QueryCache(keys_data, key_num, cache_miss_offsets, &cache_miss_cnt, cache_hit);

  // 2. Update the embedding value to the cache for cache hit keys.
  for (size_t i = 0; i < key_num; i++) {
    if (!cache_hit[i]) {
      continue;
    }
    RETURN_IF_FALSE_WITH_LOG(hash_table_->Insert(keys_data + i, 1, values_data + this->embedding_dim_ * i, nullptr),
                             "Insert hash table failed.");
  }

  if (cache_miss_cnt == 0) {
    return true;
  }

  // 3. Reserve space for cache miss keys in the cache (if there is enough space in the cache, then do nothing), write
  // the evicted element to persistent storage, and record the space in the cache, using the space in the cache first.
  RETURN_IF_FALSE_WITH_LOG(TryEvict(cache_miss_cnt), "Reserve space for miss keys failed.");

  // 4. Insert the cache miss elements into the cache from host memory.
  // Note: step 2 and step 4 can not merge.
  RETURN_IF_FALSE_WITH_LOG(InsertMissCacheFromMemory(keys_data, cache_miss_offsets, cache_miss_cnt, values_data),
                           "Insert cache miss elements into cache from host memory failed.");

  this->FreeMemory(cache_hit);
  this->FreeMemory(cache_miss_offsets);
  return true;
}

template <typename KeyType, typename ValueType, typename Allocator>
void SparseEmbeddingStorage<KeyType, ValueType, Allocator>::QueryCache(const KeyType *keys, size_t key_num,
                                                                       size_t *cache_miss_offsets,
                                                                       size_t *cache_miss_cnt, bool *cache_hit) const {
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(cache_miss_offsets);
  MS_EXCEPTION_IF_NULL(cache_miss_cnt);
  MS_EXCEPTION_IF_NULL(cache_hit);
  MS_EXCEPTION_IF_NULL(this->cache_);

  for (size_t i = 0; i < key_num; i++) {
    if (this->cache_->Exists(keys[i])) {
      // Touch keys to affect the location or order of the elements in the cache, the return value for hash table is
      // useless.
      (void)this->cache_->Get(keys[i]);
      cache_hit[i] = true;
      continue;
    }

    cache_hit[i] = false;
    // Record cache miss key's offset in all query keys.
    cache_miss_offsets[(*cache_miss_cnt)++] = i;
  }

  MS_LOG(DEBUG) << "Total keys number: " << key_num << ", cache hit number: " << (key_num - *cache_miss_cnt)
                << ", cache hit rate: " << static_cast<float>(key_num - *cache_miss_cnt) / static_cast<float>(key_num);
}

template <typename KeyType, typename ValueType, typename Allocator>
bool SparseEmbeddingStorage<KeyType, ValueType, Allocator>::TryEvict(size_t reserve_size) {
  // 1. Try evict some non-hot data in cache to reserve space for elements that will be inserted into the cache.
  if (reserve_size > this->cache_capacity_) {
    MS_LOG(EXCEPTION) << "The evict number must be less or equal to cache capacity: " << this->cache_capacity_
                      << ", but got: " << reserve_size << ", please enlarge cache capacity";
  }

  MS_EXCEPTION_IF_NULL(this->cache_);
  std::vector<CacheElement> evicted_elements;
  this->cache_->TryEvict(reserve_size, &evicted_elements);
  if (evicted_elements.size() == 0) {
    return true;
  }

  size_t evicted_keys_len = evicted_elements.size() * sizeof(KeyType);
  KeyType *evicted_keys = this->template AllocateMemory<KeyType>(evicted_keys_len);
  MS_EXCEPTION_IF_NULL(evicted_keys);

  size_t evicted_cnt = 0;
  (void)std::for_each(evicted_elements.begin(), evicted_elements.end(),
                      [&, this](const CacheElement &element) { evicted_keys[evicted_cnt++] = element.first; });

  // 2. Get all evicted embedding vector values.
  size_t evicted_values_len = evicted_elements.size() * this->embedding_dim_ * sizeof(ValueType);
  ValueType *evicted_values = this->template AllocateMemory<ValueType>(evicted_values_len);
  MS_EXCEPTION_IF_NULL(evicted_values);
  MS_EXCEPTION_IF_NULL(hash_table_);
  for (size_t i = 0; i < evicted_elements.size(); i++) {
    RETURN_IF_FALSE_WITH_LOG(
      hash_table_->Find(evicted_keys + i, 1, false, evicted_values + this->embedding_dim_ * i, nullptr),
      "Find key from hash table failed.");
    // Erase evicted element from hash table after using.
    RETURN_IF_FALSE_WITH_LOG(hash_table_->Erase(evicted_keys + i, 1, nullptr), "Erase key from hash table failed.");
  }

  if (this->cache_->size() != hash_table_->size()) {
    MS_LOG(EXCEPTION) << "The size of cache and hash table should be equal, but got cache size[" << this->cache_->size()
                      << "], hash table size[" << hash_table_->size() << "].";
  }

  // 3. Write evicted elements to persistent storage.
  MS_EXCEPTION_IF_NULL(this->storage_);
  this->storage_->Write({evicted_keys, evicted_keys_len}, {evicted_values, evicted_values_len});

  this->FreeMemory(evicted_keys);
  this->FreeMemory(evicted_values);

  return true;
}

template <typename KeyType, typename ValueType, typename Allocator>
bool SparseEmbeddingStorage<KeyType, ValueType, Allocator>::InsertMissCacheFromStorage(const KeyType *keys,
                                                                                       const size_t *cache_miss_offsets,
                                                                                       size_t cache_miss_cnt,
                                                                                       ValueType *values) {
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(cache_miss_offsets);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(this->cache_);
  MS_EXCEPTION_IF_NULL(hash_table_);

  // 1. Read the cache miss element from the persistent storage.
  size_t cache_miss_keys_len = cache_miss_cnt * sizeof(KeyType);
  KeyType *cache_miss_keys = this->template AllocateMemory<KeyType>(cache_miss_keys_len);
  MS_EXCEPTION_IF_NULL(cache_miss_keys);
  for (size_t i = 0; i < cache_miss_cnt; i++) {
    cache_miss_keys[i] = keys[cache_miss_offsets[i]];
  }
  size_t cache_miss_values_len = cache_miss_cnt * this->embedding_dim_ * sizeof(ValueType);
  ValueType *cache_miss_values = this->template AllocateMemory<ValueType>(cache_miss_values_len);
  MS_EXCEPTION_IF_NULL(cache_miss_values);

  // Read the miss values from persistent storage.
  MS_EXCEPTION_IF_NULL(this->storage_);
  this->storage_->Read({cache_miss_keys, cache_miss_keys_len}, {cache_miss_values, cache_miss_values_len});

  // 2. Insert the cache miss elements into cache, and copy them to the returned values.
  for (size_t i = 0; i < cache_miss_cnt; i++) {
    // Insert key-index pairs of the cache miss elements into the cache, the index for hash embedding table is useless,
    // set the value to 0.
    this->cache_->Put(cache_miss_keys[i], 0);

    // Insert the embedding vectors of cache miss elements to the cache.
    RETURN_IF_FALSE_WITH_LOG(
      hash_table_->Insert(cache_miss_keys + i, 1, cache_miss_values + this->embedding_dim_ * i, nullptr),
      "Insert hash table failed.");

    // Copy the embedding vectors of cache miss elements to the returned values.
    auto ret = memcpy_s(values + this->embedding_dim_ * cache_miss_offsets[i], this->embedding_dim_ * sizeof(ValueType),
                        cache_miss_values + this->embedding_dim_ * i, this->embedding_dim_ * sizeof(ValueType));
    if (ret != EOK) {
      MS_LOG(ERROR) << "Memcpy the embedding vectors of cache miss elements to the returned values failed, errno["
                    << ret << "]";
      return false;
    }
  }

  this->FreeMemory(cache_miss_keys);
  this->FreeMemory(cache_miss_values);
  return true;
}

template <typename KeyType, typename ValueType, typename Allocator>
bool SparseEmbeddingStorage<KeyType, ValueType, Allocator>::InsertMissCacheFromMemory(const KeyType *keys,
                                                                                      const size_t *cache_miss_offsets,
                                                                                      size_t cache_miss_cnt,
                                                                                      const ValueType *values) {
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(cache_miss_offsets);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(this->cache_);
  MS_EXCEPTION_IF_NULL(hash_table_);

  for (size_t i = 0; i < cache_miss_cnt; i++) {
    // Insert key-index pairs of the cache miss elements into the cache, the index for hash embedding table is useless,
    // set the value to 0.
    this->cache_->Put(keys[cache_miss_offsets[i]], 0);

    // Insert the embedding vectors of cache miss elements to the cache.
    RETURN_IF_FALSE_WITH_LOG(hash_table_->Insert(keys + cache_miss_offsets[i], 1,
                                                 values + this->embedding_dim_ * cache_miss_offsets[i], nullptr),
                             "Insert hash table failed.");
  }

  return true;
}

template class SparseEmbeddingStorage<int32_t, bool>;
template class SparseEmbeddingStorage<int32_t, int8_t>;
template class SparseEmbeddingStorage<int32_t, int16_t>;
template class SparseEmbeddingStorage<int32_t, int32_t>;
template class SparseEmbeddingStorage<int32_t, int64_t>;
template class SparseEmbeddingStorage<int32_t, uint8_t>;
template class SparseEmbeddingStorage<int32_t, uint16_t>;
template class SparseEmbeddingStorage<int32_t, uint32_t>;
template class SparseEmbeddingStorage<int32_t, uint64_t>;
template class SparseEmbeddingStorage<int32_t, float16>;
template class SparseEmbeddingStorage<int32_t, float>;
template class SparseEmbeddingStorage<int32_t, double>;

template class SparseEmbeddingStorage<int64_t, bool>;
template class SparseEmbeddingStorage<int64_t, int8_t>;
template class SparseEmbeddingStorage<int64_t, int16_t>;
template class SparseEmbeddingStorage<int64_t, int32_t>;
template class SparseEmbeddingStorage<int64_t, int64_t>;
template class SparseEmbeddingStorage<int64_t, uint8_t>;
template class SparseEmbeddingStorage<int64_t, uint16_t>;
template class SparseEmbeddingStorage<int64_t, uint32_t>;
template class SparseEmbeddingStorage<int64_t, uint64_t>;
template class SparseEmbeddingStorage<int64_t, float16>;
template class SparseEmbeddingStorage<int64_t, float>;
template class SparseEmbeddingStorage<int64_t, double>;

template class SparseEmbeddingStorage<int32_t, float, std::allocator<uint8_t>>;
template class SparseEmbeddingStorage<int64_t, float, std::allocator<uint8_t>>;
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
