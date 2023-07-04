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

#ifndef aMINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORAGE_ABSTRACT_EMBEDDING_STORAGE_H_
#define aMINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORAGE_ABSTRACT_EMBEDDING_STORAGE_H_

#include <memory>
#include <vector>

#include "include/backend/distributed/persistent/storage/storage.h"
#include "include/backend/device_address.h"

namespace mindspore {
namespace distributed {
namespace storage {
using mindspore::device::DeviceAddress;
constexpr size_t kDefaultSliceSizeInMB = 1024;

/**
 * @brief AbstractEmbeddingStorage is encapsulated within the Huge Embedding Table's lookup and update interface. It
 * supports embeddingstorage query and modification of Embeddings, interaction between the host cache(for hot spot data)
 * and persistent storage (for non-hot spot data), and preferential access to Embeddings in the host cache. If the
 * corresponding element cannot be found in the host cache, then read the element from the persistent storage.
 * Otherwise, if the host cache has insufficient space, the expired elements will automatically be evicted the to the
 * persistent storage.
 */
class AbstractEmbeddingStorage {
 public:
  AbstractEmbeddingStorage() = default;
  virtual ~AbstractEmbeddingStorage() = default;

  /**
   * @brief Initialize the embedding storage.
   * @param[in] `device_address`: The device address of the Embedding Table corresponding to the
   * AbstractEmbeddingStorage.
   */
  virtual void Initialize(const DeviceAddress *device_address) = 0;

  /**
   * @brief Finalize the AbstractEmbeddingStorage, release allocated resource.
   */
  virtual void Finalize() = 0;

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
  virtual bool Get(const ConstDataWithLen &keys, const DataWithLen &values) = 0;

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
  virtual bool Put(const ConstDataWithLen &keys, const ConstDataWithLen &values) = 0;

  /**
   * @brief To export a slice from the storage, the size is specified by the parameter 'slice_size_in_mega_bytes' in MB.
   * The default value of 1024 means that the value of the exported slice occupies 1024MB of memory.
   * @param[in] `incremental`: Determine whether export in incremental or full manner, true
   * for incremental export, false for full export
   * @param[out] `last_slice`: A bool is returned to indicate whether the slice by export is the last slice, that is,
   * the export is complete.
   * @param[in] `slice_size_in_mega_bytes`: Assign host memory in MB that the value of the exported slice occupies,
   * default 1024MB.
   * @return The byte sequence of export data.
   */
  virtual std::vector<std::shared_ptr<std::vector<char>>> ExportSlice(
    bool incremental, bool *last_slice, size_t slice_size_in_mega_bytes = kDefaultSliceSizeInMB) = 0;
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
#endif  // aMINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_STORAGE_ABSTRACT_EMBEDDING_STORAGE_H_
