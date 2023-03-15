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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORAGE_EMBEDDING_STORAGE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORAGE_EMBEDDING_STORAGE_H_

#include <memory>

#include "include/backend/distributed/embedding_cache/embedding_storage/abstract_embedding_storage.h"
#include "distributed/embedding_cache/allocator.h"
#include "distributed/embedding_cache/cache_strategy/cache.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
namespace storage {
/**
 * @brief EmbeddingStorage It is a partial implementation of AbstractEmbeddingStorage, which implements the Initialize
 * and Finalize interface and provides a general interface for allocating and deallocating memory and some common memory
 * variable.
 *
 * It is stored in key-value pair. For example, if the type of id of EmbeddingLookup is int and type of EmbeddingTable
 * is float, you can use the instance as follow: EmbeddingStorage<int, float>.
 */
template <typename KeyType, typename ValueType, typename Allocator = Allocator<uint8_t>>
class EmbeddingStorage : public AbstractEmbeddingStorage {
 public:
  // The general Allocator type used allocate host memory.
  using AllocatorType = typename std::allocator_traits<Allocator>::template rebind_alloc<uint8_t>;

  // The host cache type.
  using CacheType = Cache<KeyType, int>;

  EmbeddingStorage(int32_t embedding_key, size_t embedding_dim, size_t cache_capacity,
                   const Allocator &alloc = Allocator())
      : embedding_key_(embedding_key), embedding_dim_(embedding_dim), cache_capacity_(cache_capacity), alloc_(alloc) {}
  ~EmbeddingStorage() override = default;

  /**
   * @brief Initialize the EmbeddingStorage, such as create cache and local file storage instance.
   * @param[in] `device_address`: The device address of the Embedding Table corresponding to the EmbeddingStorage.
   */
  void Initialize(const DeviceAddress *device_address) override;

  /**
   * @brief Finalize the AbstractEmbeddingStorage, release allocated resource.
   */
  void Finalize() override;

  /**
   * @brief Batch embeddings lookup operation.
   * Query Embeddings in the host cache first, if the corresponding element cannot be found in the host cache, then read
   * the element from the persistent storage and insert host cache.
   * Access an element of the cache generally affects the location or order of the elements in the cache, depending
   * on different cache strategies.
   * @param[in] `keys`: All keys which need to query, containing data pointer and data buffer length.
   * @param[out] `values`: The output embeddings, containing data pointer and data buffer length.
   * @return Whether the function was successfully executed.
   */
  bool Get(const ConstDataWithLen &keys, const DataWithLen &values) override { return true; }

  /**
   * @brief Batch embeddings update/insert operation.
   * Update/Insert Embeddings in the host cache first, if the host cache has insufficient space, the expired elements
   * will automatically be evicted the to the persistent storage.
   * Update or Insert an element of the cache generally affects the location or order of the elements in the cache,
   * depending on different cache strategies.
   * @param[in] `keys`: All keys whose emebedding need to update, containing data pointer and data buffer length.
   * @param[in] `values`: Embeddings corresponding to all keys need to be updated, containing data pointer and data
   * buffer length.
   * @return Whether the function was successfully executed.
   */
  bool Put(const ConstDataWithLen &keys, const ConstDataWithLen &values) override { return true; }

 protected:
  /**
   * @brief Allocate host memory use alloc_.
   * @param[in] `size`: The number of bytes to allocate for memory.
   * @return The pointer to the allocated memory.
   */
  template <typename T>
  T *AllocateMemory(size_t size) {
    return reinterpret_cast<T *>(std::allocator_traits<AllocatorType>::allocate(alloc_, size));
  }

  /**
   * @brief Free host memory use alloc_.
   * @param[in] `ptr`: The pointer need to free.
   */
  void FreeMemory(void *ptr) {
    MS_EXCEPTION_IF_NULL(ptr);
    std::allocator_traits<AllocatorType>::deallocate(alloc_, reinterpret_cast<uint8_t *>(ptr), 0);
  }

  // The host cache used to record all hot spot embeddings.
  std::unique_ptr<CacheType> cache_;

  // The persistent storage(such as local file) used to record all non-hot spot embeddings.
  std::unique_ptr<StorageBase> storage_;

  // The unique key for embedding table.
  int32_t embedding_key_;

  // The embedding size of a embedding vector.
  size_t embedding_dim_;

  // The capacity of host cache for embedding storage, the same as the maximum number of key-value pairs that can be
  // saved in host cache.
  size_t cache_capacity_;

  // The common allocator used to alloacte host memory.
  AllocatorType alloc_;
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORAGE_EMBEDDING_STORAGE_H_
