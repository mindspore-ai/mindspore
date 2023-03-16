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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_DENSE_EMBEDDING_STORAGE_EMBEDDING_STORAGE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_DENSE_EMBEDDING_STORAGE_EMBEDDING_STORAGE_H_

#include <vector>
#include "distributed/embedding_cache/embedding_storage/embedding_storage.h"

namespace mindspore {
namespace distributed {
namespace storage {
/**
 * @brief A derived class for dense implementation to manage lookup and update of a huge Embedding Table for Tensor
 * type.
 */
template <typename KeyType, typename ValueType, typename Allocator = Allocator<uint8_t>>
class DenseEmbeddingStorage : public EmbeddingStorage<KeyType, ValueType, Allocator> {
 public:
  // The cache element type, a key-value pair, key is same the key of this dense embedding storage, value is the
  // index(line number in embedding table tensor) of the key.
  using CacheElement = typename EmbeddingStorage<KeyType, ValueType, Allocator>::CacheType::Element;

  DenseEmbeddingStorage(int32_t embedding_key, size_t embedding_dim, size_t cache_capacity,
                        const Allocator &alloc = Allocator())
      : EmbeddingStorage<KeyType, ValueType, Allocator>(embedding_key, embedding_dim, cache_capacity, alloc) {}
  ~DenseEmbeddingStorage() override = default;

  /**
   * @brief Initialize the EmbeddingStorage, such as recording the device address of the Embedding Table corresponding
   * to the DenseEmbeddingStorage.
   * @param[in] `device_address`: The device address of the Embedding Table tensor parameter
   * corresponding to the DenseEmbeddingStorage.
   */
  void Initialize(const DeviceAddress *device_address) override;

  /**
   * @brief Finalize the EmbeddingStorage, release allocated resource.
   */
  void Finalize() override;

  /**
   * @brief Batch embeddings lookup operation.
   * Query Embeddings in the host cache first, if the corresponding element cannot be found in the host cache, then read
   * the element from the persistent storage and insert host cache.
   * Access an element of the cache generally affects the location or order of the elements in the cache, depending
   * on different cache strategies.
   */
  bool Get(const ConstDataWithLen &keys, const DataWithLen &values) override;

  /**
   * @brief Batch embeddings update/insert operation.
   * Update/Insert Embeddings in the host cache first, if the host cache has insufficient space, the expired elements
   * will automatically be evicted the to the persistent storage.
   * Update or Insert an element of the cache generally affects the location or order of the elements in the cache,
   * depending on different cache strategies.
   */
  bool Put(const ConstDataWithLen &keys, const ConstDataWithLen &values) override;

 private:
  /**
   * @brief Query cache to get the index in the embedding table tensor of each cache hit key, and count the number of
   * cache miss keys. Access an element of the cache generally affects the location or order of the elements in the
   * cache, depending on different cache strategies.
   * @param[in] `keys`: The array records all keys which need to query.
   * @param[in] `key_num`: The number of keys which need to query.
   * @param[out] `cache_miss_offsets`: The array records the offset(index) of cache miss key in origin keys array.
   * @param[out] `cache_miss_cnt`: The number of cache miss keys.
   * @param[out] `indices_in_cache`: The array records the indices in the embedding table tensor of each cache hit
   * keys.
   */
  void QueryCache(const KeyType *keys, size_t key_num, int *cache_miss_offsets, size_t *cache_miss_cnt,
                  int *indices_in_cache) const;

  /**
   * @brief Reserve space for cache miss keys in the cache, write the evicted element to persistent storage,
   * and record the new space position in the cache.
   * @param[in] `reserve_size`: The number of element slots that are expected to be reserved. If the
   * reserve_size is less than or equal to the number of slots remaining in the cache, the function does nothing.
   * @return Whether the function was successfully executed.
   */
  bool TryEvict(size_t reserve_size);

  /**
   * @brief Insert the cache miss elements into the cache from persistent storage, and copy them to the output values.
   * @param[in] `keys`: The array records all origin keys for batch embeddings lookup operation.
   * @param[in] `cache_miss_offsets`: The array records the offset(index) of cache miss key in origin keys array.
   * @param[in] `cache_miss_cnt`: The number of cache miss keys.
   * @param[out] `values`: The output embeddings.
   * @return Whether the function was successfully executed.
   */
  bool InsertMissCacheFromStorage(const KeyType *keys, const int *cache_miss_offsets, size_t cache_miss_cnt,
                                  ValueType *values);

  /**
   * @brief Insert the cache miss elements into the cache from host memory.
   * @param[in] `keys`: The array records all origin keys for batch embeddings update/insert operation.
   * @param[in] `cache_miss_offsets`: The array records the offset(index) of cache miss key in origin keys array.
   * @param[in] `cache_miss_cnt`: The number of cache miss keys.
   * @param[in] `values`: Embeddings corresponding to all keys need to be updated.
   * @return Whether the function was successfully executed.
   */
  bool InsertMissCacheFromMemory(const KeyType *keys, const int *cache_miss_offsets, size_t cache_miss_cnt,
                                 const ValueType *values);

  // The base pointer to embedding table parameter, all embeddings in host cache is recorded in
  // embedding_param_address_.
  const DeviceAddress *embedding_param_address_{nullptr};

  // For performance, keep the pointer snapshot for `embedding_param_address_`.
  ValueType *embedding_param_ptr_{nullptr};

  // Record all empty slot(idle slot or index) in embedding table tensor.
  std::vector<int> empty_slots_;
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_DENSE_EMBEDDING_STORAGE_EMBEDDING_STORAGE_H_
