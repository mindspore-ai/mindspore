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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_CHCHE_UTILS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_CHCHE_UTILS_H_

#include <map>
#include <string>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "kernel/kernel.h"
#include "distributed/embedding_cache/embedding_hash_map.h"
#include "distributed/embedding_cache/embedding_store.h"
#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
class EmbeddingCachePrefetchActor;
}  // namespace runtime

namespace distributed {
// The local host cache size defaults to 10 times the device cache size.
static constexpr size_t kHostCacheScaleFactor = 10;
// The maximum number of concurrent threads for data prefetching.
static constexpr size_t kMaxThreadNum = 16;
// Maximum number of feature ids processed per thread.
static constexpr size_t kMaxIdsPerThread = 10000;

static constexpr size_t kNumberBase = 10;

static constexpr size_t kOneGBBitNum = 30;

// The default cache size of one embedding parameter on role of server: 100GB.
static constexpr size_t kDefaultEmbeddingRemoteCacheMemorySize = size_t(100) << 30;

// The default cache size of one embedding parameter on role of worker: 10GB.
static constexpr size_t kDefaultEmbeddingLocalCacheMemorySize = size_t(10) << 30;

static constexpr auto kEnvEmbeddingRemoteCacheMemorySize = "MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE";

static constexpr auto kEnvEmbeddingLocalCacheMemorySize = "MS_EMBEDDING_LOCAL_CACHE_MEMORY_SIZE";

using mindspore::kernel::Address;

// The hash tables records information such as the dimension, memory address, and cache size of the embedding table
// with the embedding cache enabled.
struct HashTableInfo {
  size_t cache_vocab_size{0};
  size_t host_cache_vocab_size{0};
  size_t embedding_size{0};
  size_t vocab_size{0};
  Address device_address{nullptr, 0};
  std::shared_ptr<float> host_address{nullptr};
  int32_t param_key_{-1};
};

// Record the hash mapping relationship of all embedding tables with cache enabled on the device side, and the
// ids information that needs to be exchanged with the local host cache. Note that the following information of
// all embedding cache tables on the device side is same: hash mapping, and feature ids of feature vectors that need
// to be swapped with the local host cache.
struct EmbeddingDeviceCache {
  EmbeddingDeviceCache(size_t batch_ids_num, size_t cache_vocab_size)
      : hash_swap_index_addr_(nullptr), hash_swap_value_addr_(nullptr) {
    device_to_host_index = std::make_unique<int[]>(batch_ids_num);
    device_to_host_ids = std::make_unique<int[]>(batch_ids_num);
    host_to_device_index = std::make_unique<int[]>(batch_ids_num);
    host_to_device_ids = std::make_unique<int[]>(batch_ids_num);
    device_hash_map_ = std::make_shared<EmbeddingHashMap>(0, cache_vocab_size);
  }

  std::unique_ptr<int[]> device_to_host_index;
  std::unique_ptr<int[]> device_to_host_ids;
  std::unique_ptr<int[]> host_to_device_index;
  std::unique_ptr<int[]> host_to_device_ids;
  int *hash_swap_index_addr_;
  float *hash_swap_value_addr_;
  std::shared_ptr<EmbeddingHashMap> device_hash_map_;
};

// Record the hash mapping relationship of all embedding tables with cache enabled on the local host side, and the
// information that needs to be exchanged with the remote cache and device cache. Note that the following information of
// all embedding cache tables on the local host side is same: hash mapping, and feature ids of feature vectors that need
// to be swapped with the remote cache and device cache.
struct EmbeddingHostCache {
  EmbeddingHostCache(size_t batch_ids_num, size_t host_cache_vocab_size) {
    host_to_server_index = std::make_unique<int[]>(batch_ids_num);
    host_to_server_ids = std::make_unique<int[]>(batch_ids_num);
    server_to_host_index = std::make_unique<int[]>(batch_ids_num);
    server_to_host_ids = std::make_unique<int[]>(batch_ids_num);
    host_to_device_index = std::make_unique<int[]>(batch_ids_num);
    device_to_host_index = std::make_unique<int[]>(batch_ids_num);
    host_hash_map_ = std::make_shared<EmbeddingHashMap>(0, host_cache_vocab_size);
  }

  std::unique_ptr<int[]> host_to_server_index;
  std::unique_ptr<int[]> host_to_server_ids;
  std::unique_ptr<int[]> server_to_host_index;
  std::unique_ptr<int[]> server_to_host_ids;
  std::unique_ptr<int[]> host_to_device_index;
  std::unique_ptr<int[]> device_to_host_index;
  std::shared_ptr<EmbeddingHashMap> host_hash_map_;
};

struct EmbeddingCacheStatisticsInfo {
  size_t batch_id_count_{0};
  size_t batch_id_unique_count_{0};
  size_t device_to_host_size_{0};
  size_t host_to_device_size_{0};
  size_t host_to_server_size_{0};
  size_t server_to_host_size_{0};
  size_t hash_hit_count_{0};
  size_t mem_cache_swap_out_size_{0};
  size_t mem_cache_swap_in_size_{0};
  size_t mem_cache_hit_count_{0};
};

// The EmbeddingCacheTableManager class is used to save all Parameter information for enabling cache, such as device
// cache size, host cache size, etc., and can allocate memory for the embedding cache table.
class BACKEND_EXPORT EmbeddingCacheTableManager {
 public:
  static EmbeddingCacheTableManager &GetInstance();

  // Initialize the EmbeddingCacheTableManager.
  void Initialize();
  // Finalize the EmbeddingCacheTableManager and release all resource.
  void Finalize();

  // Insert and save dimension information of the embedding cache table.
  void InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                           size_t vocab_size, int32_t param_key);

  // Parameter will modify the name. After modification, you need to re-insert all the dimension information that saves
  // the parameter.
  void ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name,
                             size_t cache_vocab_size, size_t embedding_size);

  // Clone a hash table, such as the optimizer's state parameters are generally cloned from weight.
  void CloneHashTable(const std::string &dest_param_name, int32_t dest_param_key, const std::string &src_param_name,
                      int32_t src_param_key);

  // Alloc device memory for all embedding cache table.
  void AllocMemForEmbeddingCacheTable(const device::DeviceContext *device_context);

  // Qeury device address of a embedding cache table.
  const Address &QueryHashTableAddr(const std::string &param_name) const;

  // Qeury device cache size of a embedding cache table.
  size_t QueryHashTableSize(const std::string &param_name) const;

  // Check whether a parameter is cache enabled embedding table.
  bool IsEmbeddingCacheTable(const std::string &param_name) const { return hash_tables_.count(param_name) != 0; }

  // Set ids number of a batchsize.
  void set_batch_ids_num(size_t batch_ids_num) { batch_ids_num_ = batch_ids_num; }

  //  Get the offset of the id range corresponding to the embedding cache table slice on each worker in a multi-worker
  //  automatic parallel scenario.
  int cache_indices_lower_bound() const;

  // Get vocab cache size on device.
  size_t vocab_cache_size() const { return device_cache_size_; }

  void DumpHashTables() const;

 private:
  EmbeddingCacheTableManager() = default;
  ~EmbeddingCacheTableManager() = default;
  DISABLE_COPY_AND_ASSIGN(EmbeddingCacheTableManager);

  // Get embedding table slice bound info on each worker in a multi-worker automatic parallel scenario.
  void GetEmbeddingTableSliceBound();

  // The hash tables records information such as the dimension, memory address, and cache size of the embedding table
  // with the embedding cache enabled.
  std::map<std::string, HashTableInfo> hash_tables_;

  // Record the hash mapping relationship of all embedding tables with cache enabled on the device side, and the
  // ids information that needs to be exchanged with the local host cache.
  std::shared_ptr<EmbeddingDeviceCache> embedding_device_cache_;

  // Record the hash mapping relationship of all embedding tables with cache enabled on the local host side, and the
  // information that needs to be exchanged with the remote cache and device cache.
  std::shared_ptr<EmbeddingHostCache> embedding_host_cache_;

  // Model parallelism is used between multiple workers, and local_embedding_slice_bounds_ records the feature range
  // corresponding to the embedding table slice of the process.
  std::pair<int, int> local_embedding_slice_bounds_;

  // Model parallelism is used between multiple workers, and local_device_cache_bounds_ records the local device cache
  // range corresponding to the embedding table slice of the process.
  std::pair<int, int> local_device_cache_bounds_;

  // Full Embedding table row num, not less than the total number of feature ids.
  size_t vocab_size_{0};
  // Embedding cache size(row number of embedding cache) of device cache.
  size_t device_cache_size_{0};
  // Embedding cache size(row number of embedding cache) of local host cache.
  size_t host_cache_size_{0};
  // Total ids number of a batchsize.
  size_t batch_ids_num_{0};

  friend class mindspore::runtime::EmbeddingCachePrefetchActor;
};

// The EmbeddingStoreManager class is used to save all enabling store for all embedding table.
class BACKEND_EXPORT EmbeddingStoreManager {
 public:
  static EmbeddingStoreManager &GetInstance();

  void Add(const std::string &name, std::shared_ptr<EmbeddingStore<size_t, float_t>> emb_store) {
    embedding_stores_[name] = emb_store;
  }
  std::shared_ptr<EmbeddingStore<size_t, float_t>> Get(const std::string &name) { return embedding_stores_[name]; }

 private:
  EmbeddingStoreManager() = default;
  ~EmbeddingStoreManager() = default;
  DISABLE_COPY_AND_ASSIGN(EmbeddingStoreManager);

  mindspore::HashMap<std::string, std::shared_ptr<EmbeddingStore<size_t, float_t>>> embedding_stores_;
};

size_t GetEmbeddingRemoteCacheSize();

size_t GetEmbeddingLocalCacheSize();
}  // namespace distributed
static distributed::EmbeddingCacheTableManager &embedding_cache_table_manager =
  distributed::EmbeddingCacheTableManager::GetInstance();

static distributed::EmbeddingStoreManager &embedding_store_manager = distributed::EmbeddingStoreManager::GetInstance();
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_CHCHE_UTILS_H_
