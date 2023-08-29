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

#include <future>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <utility>
#include "kernel/kernel.h"
#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"
#include "include/backend/distributed/embedding_cache/embedding_storage/abstract_embedding_storage.h"
#include "include/backend/distributed/embedding_cache/embedding_hash_map.h"
#include "include/backend/distributed/embedding_cache/blocking_queue.h"
#include "include/backend/data_queue/data_queue.h"

namespace mindspore {
namespace runtime {
class EmbeddingCachePrefetchActor;
class DeviceEmbeddingOperation;
class DeviceDenseEmbeddingOperation;
class DeviceSparseEmbeddingOperation;
}  // namespace runtime

namespace distributed {
// The local host cache size defaults to 10 times the device cache size.
static constexpr size_t kHostCacheScaleFactor = 10;
// The maximum number of concurrent threads for data prefetching.
static constexpr size_t kMaxThreadNum = 16;
// Maximum number of feature ids processed per thread.
static constexpr size_t kMaxIdsPerThread = 10000;

// Prefetch 16 batchs data once.
static constexpr size_t kMultiBatchThreshold = 16;

using mindspore::device::DeviceAddress;
using mindspore::kernel::Address;

// The type of embedding tables.
enum ParamType { kUnKnown = 0, kWeight = 1, kAccumulation = 2 };

// The initialization information for embedding tables.
struct ParamInitInfo {
  std::string param_name_;
  ParamType param_type_{kUnKnown};
  size_t global_seed_{0};
  size_t op_seed_{0};
  float init_val_{0};
};

// The hash tables records information such as the dimension, memory address, and cache size of the embedding table
// with the embedding cache enabled.
struct HashTableInfo {
  size_t cache_vocab_size{0};
  size_t host_cache_vocab_size{0};
  size_t embedding_size{0};
  size_t vocab_size{0};
  // For performance, set address the snapshot of device_address.
  Address address{nullptr, 0};
  DeviceAddress *device_address{nullptr};
  float *host_address{nullptr};
  ParamInitInfo param_init_info_;
  int32_t param_key_{-1};
};

// Record the hash mapping relationship of all embedding tables with cache enabled on the device side, and the
// ids information that needs to be exchanged with the local host cache. Note that the following information of
// all embedding cache tables on the device side is same: hash mapping, and feature ids of feature vectors that need
// to be swapped with the local host cache.
struct EmbeddingDeviceCache {
  explicit EmbeddingDeviceCache(size_t batch_ids_num);

  std::unique_ptr<int[]> device_to_host_index;
  std::unique_ptr<int[]> device_to_host_ids;
  std::unique_ptr<int[]> host_to_device_index;
  std::unique_ptr<int[]> host_to_device_ids;
};

// Record the hash mapping relationship of all embedding tables with cache enabled on the local host side, and the
// information that needs to be exchanged with the remote cache and device cache. Note that the following information of
// all embedding cache tables on the local host side is same: hash mapping, and feature ids of feature vectors that need
// to be swapped with the remote cache and device cache.
struct EmbeddingHostCache {
  explicit EmbeddingHostCache(size_t batch_ids_num);

  std::unique_ptr<int[]> host_to_server_index;
  std::unique_ptr<int[]> host_to_server_ids;
  std::unique_ptr<int[]> server_to_host_index;
  std::unique_ptr<int[]> server_to_host_ids;
  std::unique_ptr<int[]> new_id_index;
  std::unique_ptr<int[]> host_to_device_index;
  std::unique_ptr<int[]> device_to_host_index;
};

struct EmbeddingCacheStatisticsInfo {
  size_t batch_id_count_{0};
  size_t batch_id_unique_count_{0};
  size_t device_to_host_size_{0};
  size_t host_to_device_size_{0};
  size_t host_to_server_size_{0};
  size_t server_to_host_size_{0};
  size_t new_id_size_{0};
  size_t hash_hit_count_{0};
  size_t mem_cache_swap_out_size_{0};
  size_t mem_cache_swap_in_size_{0};
  size_t mem_cache_hit_count_{0};
};

// Origin id data item recorder.
struct IdDataInfo {
  IdDataInfo() = default;
  IdDataInfo(void *data, size_t size, std::vector<device::DataQueueItem> *items, bool end_of_epoch, bool end_of_file)
      : data_(data), size_(size), items_(items), end_of_epoch_(end_of_epoch), end_of_file_(end_of_file) {}

  void *data_{nullptr};
  size_t size_{0};
  std::vector<device::DataQueueItem> *items_{nullptr};
  bool end_of_epoch_{false};
  bool end_of_file_{false};
};

// The indexes data after cache prefetch.
struct IndexDataInfo {
  IndexDataInfo() = default;
  IndexDataInfo(void *data, std::vector<device::DataQueueItem> *items, bool end_of_epoch, bool end_of_file)
      : data_(data), items_(items), end_of_epoch_(end_of_epoch), end_of_file_(end_of_file) {}

  void *data_{nullptr};
  std::vector<device::DataQueueItem> *items_{nullptr};
  bool end_of_epoch_{false};
  bool end_of_file_{false};
};

// The origin unique data recorder.
struct UniqueIds {
  UniqueIds() = default;

  size_t data_step_{0};
  std::vector<void *> multi_batch_data_;
  std::vector<size_t> multi_batch_size_;
  std::vector<std::vector<device::DataQueueItem> *> multi_batch_items_;
  int *ids_{nullptr};
  size_t ids_num_{0};

  bool end_of_epoch_{false};
  bool end_of_file_{false};
};

// Record all information used to analyse cache.
struct CacheAnalysis {
  CacheAnalysis() = default;
  CacheAnalysis(EmbeddingDeviceCache *embedding_device_cache, EmbeddingHostCache *embedding_host_cache,
                EmbeddingCacheStatisticsInfo *statistics_info, UniqueIds *unique_ids, int *indices, bool end_of_epoch,
                bool end_of_file)
      : embedding_device_cache_(embedding_device_cache),
        embedding_host_cache_(embedding_host_cache),
        statistics_info_(statistics_info),
        unique_ids_(unique_ids),
        indices_(indices),
        end_of_epoch_(end_of_epoch),
        end_of_file_(end_of_file) {}

  // Record the ids information that needs to be exchanged with the local host cache.
  EmbeddingDeviceCache *embedding_device_cache_{nullptr};
  // Record the information that needs to be exchanged with the remote cache and device cache.
  EmbeddingHostCache *embedding_host_cache_{nullptr};
  EmbeddingCacheStatisticsInfo *statistics_info_{nullptr};
  UniqueIds *unique_ids_{nullptr};
  int *indices_{nullptr};
  bool end_of_epoch_{false};
  bool end_of_file_{false};
};

// Record all ids(after unique) and indices(after cache analysis)
struct IdsAndIndices {
  IdsAndIndices() = default;
  IdsAndIndices(UniqueIds *unique_ids, int *indices, bool end_of_epoch, bool end_of_file)
      : unique_ids_(unique_ids), indices_(indices), end_of_epoch_(end_of_epoch), end_of_file_(end_of_file) {}

  UniqueIds *unique_ids_{nullptr};
  int *indices_{nullptr};
  bool end_of_epoch_{false};
  bool end_of_file_{false};
};

// The EmbeddingCacheTableManager class is used to save all Parameter information for enabling cache, such as device
// cache size, host cache size, etc., and can allocate memory for the embedding cache table.
class BACKEND_EXPORT EmbeddingCacheTableManager {
 public:
  using WarmUpCacheMapValue = std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr>;
  using WarmUpCacheMapEntry = std::pair<int32_t, WarmUpCacheMapValue>;
  using WarmUpCacheMap = std::map<int32_t, WarmUpCacheMapValue>;
  static EmbeddingCacheTableManager &GetInstance();

  // Initialize the EmbeddingCacheTableManager.
  void Initialize();
  // Finalize the EmbeddingCacheTableManager and release all resource.
  void Finalize(const device::DeviceContext *device_context);

  // Insert and save dimension information of the embedding cache table.
  void InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                           size_t vocab_size, int32_t param_key);

  // Parameter will modify the name. After modification, you need to re-insert all the dimension information that saves
  // the parameter.
  void ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name);

  // Insert the initial value for the accumulation value of embedding's optimizer.
  void InsertAccumuInitInfo(const std::string &param_name, float init_val);

  // Clone a hash table, such as the optimizer's state parameters are generally cloned from weight.
  void CloneHashTable(const std::string &dest_param_name, int32_t dest_param_key, const std::string &src_param_name,
                      int32_t src_param_key);

  // Set the device address for embedding cache table, using the same device address with parameter node.
  void SetEmbeddingDeviceAddress(const std::string &param_name, DeviceAddress *device_address);

  // Alloc device memory for all embedding cache table.
  void AllocMemForEmbedding(const device::DeviceContext *device_context);

  // Qeury device address of a embedding cache table.
  const DeviceAddress *QueryEmbeddingDeviceAddress(const std::string &param_name) const;

  // Qeury device cache size of a embedding cache table.
  size_t QueryHashTableSize(const std::string &param_name) const;

  // Check whether a parameter is cache enabled embedding table.
  bool IsEmbeddingCacheTable(const std::string &param_name) const { return hash_tables_.count(param_name) != 0; }

  // Set ids number of a batchsize.
  void set_batch_ids_num(size_t batch_ids_num) { batch_ids_num_ = batch_ids_num; }

  //  Get the offset of the id range corresponding to the embedding cache table slice on each worker in a multi-worker
  //  automatic parallel scenario.
  int cache_indices_lower_bound() const;

  // Set embedding vocab cache size on device.
  void set_cache_size(size_t cache_size) { device_cache_size_ = cache_size; }

  // Get embedding vocab cache size on device.
  size_t cache_size() const { return device_cache_size_; }

  // Set the storage format (`dense` or `sparse`) of embedding tables.
  void set_sparse_format(bool is_sparse) { sparse_format_ = is_sparse; }

  bool is_sparse_format() { return sparse_format_; }

  // Get whether multi-stage pipeline cache prefetch is enabled.
  bool enable_pipeline() const;

  void DumpHashTables() const;

  bool checkpoint_load_status() const { return checkpoint_load_status_; }

  void set_checkpoint_load_status(bool checkpoint_load_status) { checkpoint_load_status_ = checkpoint_load_status; }

  int32_t StoreWarmUpPtr(const int32_t param_key, const tensor::TensorPtr &tensor_ptr);

  int32_t StoreWarmUpPtr(const int32_t param_key, const tensor::TensorPtr &key_ptr, const tensor::TensorPtr &value_ptr,
                         const tensor::TensorPtr &status_ptr);

  void WarmUpHostCacheAsync(const int32_t batch_count);

  std::pair<std::shared_ptr<std::future<bool>>, bool> GetWarmUpHostCacheAsyncStatus();

  bool WaitForWarmUpHostCacheComplete();

  const HashTableInfo *FindHashTablesByParamKey(const int param_key);

  const WarmUpCacheMap &host_cache_ptrs() { return host_cache_ptrs_; }

  std::map<std::string, HashTableInfo> &hash_tables() { return hash_tables_; }

  void set_host_hash_map(const std::shared_ptr<EmbeddingHashMap> &host_hash_map) { host_hash_map_ = host_hash_map; }

 private:
  EmbeddingCacheTableManager() = default;
  ~EmbeddingCacheTableManager() = default;
  DISABLE_COPY_AND_ASSIGN(EmbeddingCacheTableManager);

  // Get embedding table slice bound info on each worker in a multi-worker automatic parallel scenario.
  void GetEmbeddingTableSliceBound();

  void WarmUpHostCacheItemBatch(const int32_t thread_count, const WarmUpCacheMapEntry &entry);

  void WarmUpHostCacheItem(const std::shared_ptr<EmbeddingHashMap> &embedding_hash_map,
                           const HashTableInfo *hash_table_info_ptr, const WarmUpCacheMapEntry &entry, const int start,
                           const int end, const size_t value_len);

  void WarmUpHostCacheSync(const int32_t batch_count);

  std::atomic<bool> checkpoint_load_status_{false};

  WarmUpCacheMap host_cache_ptrs_;

  std::mutex host_cache_mutex_;

  std::shared_ptr<std::promise<bool>> host_cache_promise_{nullptr};

  // The hash tables records information such as the dimension, memory address, and cache size of the embedding table
  // with the embedding cache enabled.
  std::map<std::string, HashTableInfo> hash_tables_;

  std::shared_ptr<EmbeddingHashMap> device_hash_map_;

  std::shared_ptr<EmbeddingHashMap> host_hash_map_;

  int *hash_swap_index_addr_;
  float *hash_swap_value_addr_;

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

  // If the storage format is sparse or dense, the default format is dense.
  bool sparse_format_{false};

  // The batch number once cache prefetch.
  size_t multi_batch_threshold_;

  // Record whether multi-stage pipeline cache prefetch is enabled.
  bool enable_pipeline_{false};

  device::DeviceContext *cpu_device_context_{nullptr};

  friend class mindspore::runtime::EmbeddingCachePrefetchActor;
  friend class mindspore::runtime::DeviceEmbeddingOperation;
  friend class mindspore::runtime::DeviceDenseEmbeddingOperation;
  friend class mindspore::runtime::DeviceSparseEmbeddingOperation;
};

/**
 * @brief A single instance class used to manager all EmbeddingStorage instances, EmbeddingStorage is encapsulated
 * within the Huge Embedding Table's lookup and update. EmbeddingStorageManager provides Add and Get API to add, replace
 * and acquire EmbeddingStorage instances.
 */
class BACKEND_EXPORT EmbeddingStorageManager {
 public:
  static EmbeddingStorageManager &GetInstance();

  /**
   * @brief Add the embedding storage instance corresponding to the parameter key, if embedding storage instance already
   * exists, replace it by input parameter `embed_storage'.
   * @param[in] `param_key`: The parameter key for embedding table which need to add.
   * @param[in] `embed_storage`: The embedding storage instance pointer which can not be nullptr.
   */
  void Add(int32_t param_key, const std::shared_ptr<storage::AbstractEmbeddingStorage> &embed_storage);

  /**
   * @brief Try get the embedding storage instance corresponding to the parameter key.
   * @param[in] `param_key`: The parameter key for embedding table which need to acquire.
   * @return The embedding storage instance pointer if the embedding storage already exists, else throw exception.
   */
  std::shared_ptr<storage::AbstractEmbeddingStorage> Get(int32_t param_key);

  /**
   * @brief Check if the embedding storage instance corresponding to the parameter key already exists.
   * @param[in] `param_key`: The parameter key for embedding table which need to check if the embedding storage already
   * exists.
   * @return true if the embedding storage already exists, else false.
   */
  bool Exists(int32_t param_key) const { return embedding_storages_.find(param_key) != embedding_storages_.end(); }

  /**
   * @brief Clear all embedding storage instances and release related resources.
   */
  void Clear();

 private:
  EmbeddingStorageManager() = default;
  ~EmbeddingStorageManager() = default;
  DISABLE_COPY_AND_ASSIGN(EmbeddingStorageManager);

  // Record all {parameter key -> embedding storage instance} pairs.
  HashMap<int32_t, std::shared_ptr<storage::AbstractEmbeddingStorage>> embedding_storages_;
};

/**
 * @brief Create a new embedding storage instance for specific key and value type, and add the instance to
 * EmbeddingStorageManager.
 * @param[in] `key_value_types`: The specific key and value data type to determine the type of embedding storage
 * instance to create.
 * @param[in] `embedding_key`: The unique parameter key for embedding table.
 * @param[in] `embedding_dim`: The size of each embedding vector.
 * @param[in] `capacity`: The capacity for new embedding storage.
 */
BACKEND_EXPORT void CreateEmbeddingStorage(std::pair<TypeId, TypeId> key_value_types, int32_t embedding_key,
                                           size_t embedding_dim, size_t capacity);
}  // namespace distributed

static distributed::EmbeddingCacheTableManager &embedding_cache_table_manager =
  distributed::EmbeddingCacheTableManager::GetInstance();

static distributed::EmbeddingStorageManager &embedding_storage_manager =
  distributed::EmbeddingStorageManager::GetInstance();
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_CHCHE_UTILS_H_
