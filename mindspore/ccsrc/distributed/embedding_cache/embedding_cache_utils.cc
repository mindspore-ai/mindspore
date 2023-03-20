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

#include "include/backend/distributed/embedding_cache/embedding_cache_utils.h"
#include <algorithm>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
#include "include/backend/distributed/cluster/cluster_context.h"
#endif
#include "include/backend/distributed/ps/ps_context.h"
#include "distributed/embedding_cache/embedding_storage/dense_embedding_storage.h"
#include "distributed/embedding_cache/embedding_storage/sparse_embedding_storage.h"
#include "include/backend/distributed/embedding_cache/embedding_storage/abstract_embedding_storage.h"

namespace mindspore {
namespace distributed {
EmbeddingCacheTableManager &EmbeddingCacheTableManager::GetInstance() {
  static EmbeddingCacheTableManager instance{};
  return instance;
}

void EmbeddingCacheTableManager::Initialize() { GetEmbeddingTableSliceBound(); }

void EmbeddingCacheTableManager::Finalize() {
  hash_tables_.clear();

  embedding_device_cache_ = nullptr;
  embedding_host_cache_ = nullptr;
}

void EmbeddingCacheTableManager::InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size,
                                                     size_t embedding_size, size_t vocab_size, int32_t param_key) {
  if (cache_vocab_size == 0 || embedding_size == 0 || vocab_size == 0) {
    MS_LOG(EXCEPTION) << "The size of hash table can not equal to zero.";
  }
  hash_tables_[param_name].cache_vocab_size = cache_vocab_size;
  hash_tables_[param_name].host_cache_vocab_size = cache_vocab_size * kHostCacheScaleFactor;
  hash_tables_[param_name].embedding_size = embedding_size;
  hash_tables_[param_name].vocab_size = vocab_size;
  hash_tables_[param_name].param_key_ = param_key;

  if (vocab_size_ == 0) {
    vocab_size_ = vocab_size;
  }
  if (device_cache_size_ == 0) {
    device_cache_size_ = cache_vocab_size;
  }
  if (host_cache_size_ == 0) {
    host_cache_size_ = cache_vocab_size * kHostCacheScaleFactor;
  }
}

void EmbeddingCacheTableManager::ReInsertHashTableSize(const std::string &new_param_name,
                                                       const std::string &cur_param_name) {
  if (new_param_name.empty() || cur_param_name.empty()) {
    MS_LOG(EXCEPTION) << "Parameter name can not be empty.";
  }
  if (new_param_name == cur_param_name) {
    return;
  }
  auto iter = hash_tables_.find(cur_param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find parameter[" << cur_param_name << "] in hash table.";
  }
  (void)hash_tables_.emplace(new_param_name, iter->second);
  (void)hash_tables_.erase(iter);
}

void EmbeddingCacheTableManager::InsertAccumuInitInfo(const std::string &param_name, float init_val) {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find parameter[" << param_name << "] in hash table.";
  }
  auto &hash_table_info = iter->second;
  if (hash_table_info.param_init_info_.param_type_ != kUnKnown) {
    return;
  }
  MS_LOG(INFO) << "Insert accumulation init info:" << param_name << ", init value:" << init_val;
  hash_table_info.param_init_info_.param_name_ = param_name;
  hash_table_info.param_init_info_.param_type_ = kAccumulation;
  hash_table_info.param_init_info_.init_val_ = init_val;
}

void EmbeddingCacheTableManager::CloneHashTable(const std::string &dest_param_name, int32_t dest_param_key,
                                                const std::string &src_param_name, int32_t src_param_key) {
  if (dest_param_name == src_param_name) {
    MS_LOG(INFO) << "The dest_param_name is same as src_param_name";
    return;
  }
  auto iter = hash_tables_.find(src_param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "The source hash table[" << src_param_name << "] does not exist, clone failed.";
  }
  (void)hash_tables_.emplace(dest_param_name, iter->second);
  hash_tables_[src_param_name].param_key_ = src_param_key;
  hash_tables_[dest_param_name].param_key_ = dest_param_key;
}

const DeviceAddress *EmbeddingCacheTableManager::QueryEmbeddingDeviceAddress(const std::string &param_name) const {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find device address of " << param_name;
  }
  return iter->second.device_address;
}

size_t EmbeddingCacheTableManager::QueryHashTableSize(const std::string &param_name) const {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find vocab cache size of " << param_name;
  }
  return iter->second.cache_vocab_size;
}

void EmbeddingCacheTableManager::SetEmbeddingDeviceAddress(const std::string &param_name,
                                                           DeviceAddress *device_address) {
  MS_EXCEPTION_IF_NULL(device_address);
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find hash table info for " << param_name;
  }
  iter->second.device_address = device_address;
}

void EmbeddingCacheTableManager::AllocMemForEmbedding(const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  size_t max_embedding_size = 0;
  for (auto &item : hash_tables_) {
    auto *device_address = item.second.device_address;
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr) {
      device_context->device_res_manager_->AllocateMemory(device_address);
    }
    item.second.address = Address(device_address->GetMutablePtr(), device_address->GetSize());

    size_t embedding_size = item.second.embedding_size;
    auto &host_address = item.second.host_address;
    auto host_hash_table_addr = std::make_unique<float[]>(host_cache_size_ * embedding_size);
    MS_EXCEPTION_IF_NULL(host_hash_table_addr);
    host_address = std::shared_ptr<float>(host_hash_table_addr.release(), std::default_delete<float[]>());
    MS_EXCEPTION_IF_NULL(host_address);

    max_embedding_size = (embedding_size > max_embedding_size) ? embedding_size : max_embedding_size;
  }

  embedding_device_cache_ = std::make_shared<EmbeddingDeviceCache>(batch_ids_num_, device_cache_size_);
  MS_EXCEPTION_IF_NULL(embedding_device_cache_);
  embedding_host_cache_ = std::make_shared<EmbeddingHostCache>(batch_ids_num_, host_cache_size_);
  MS_EXCEPTION_IF_NULL(embedding_host_cache_);

  embedding_device_cache_->hash_swap_index_addr_ =
    reinterpret_cast<int *>(device_context->device_res_manager_->AllocateMemory(batch_ids_num_ * sizeof(int)));
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->hash_swap_index_addr_);
  embedding_device_cache_->hash_swap_value_addr_ = reinterpret_cast<float *>(
    device_context->device_res_manager_->AllocateMemory(max_embedding_size * batch_ids_num_ * sizeof(float)));
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->hash_swap_value_addr_);
}

void EmbeddingCacheTableManager::GetEmbeddingTableSliceBound() {
  auto worker_num = ps::PSContext::instance()->worker_num();
  if (worker_num == 0) {
    return;
  }

  uint32_t rank_id = 0;
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  auto node = distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  rank_id = node->rank_id();
#endif

  auto local_shard_size = FloatToInt(std::ceil(SizeToFloat(vocab_size_) / worker_num));
  local_embedding_slice_bounds_.first = local_shard_size * UintToInt(rank_id);
  local_embedding_slice_bounds_.second =
    std::min(local_embedding_slice_bounds_.first + local_shard_size, SizeToInt(vocab_size_));
  local_device_cache_bounds_.first = SizeToInt(device_cache_size_) * UintToInt(rank_id);
  local_device_cache_bounds_.second = local_device_cache_bounds_.first + SizeToInt(device_cache_size_);
  MS_LOG(INFO) << "Worker num:" << worker_num << ", rank id:" << rank_id
               << ", id begin:" << local_embedding_slice_bounds_.first
               << ", id end:" << local_embedding_slice_bounds_.second
               << ", cache indices begin: " << local_device_cache_bounds_.first
               << ", cache indices end: " << local_device_cache_bounds_.second;
}

int EmbeddingCacheTableManager::cache_indices_lower_bound() const { return local_device_cache_bounds_.first; }

void EmbeddingCacheTableManager::DumpHashTables() const {
  MS_EXCEPTION_IF_NULL(embedding_device_cache_);
  for (const auto &item : hash_tables_) {
    const auto &param_name = item.first;
    size_t cache_vocab_size = item.second.cache_vocab_size;
    size_t host_cache_vocab_size = item.second.host_cache_vocab_size;
    size_t embedding_size = item.second.embedding_size;
    size_t vocab_size = item.second.vocab_size;
    int32_t param_key = item.second.param_key_;
    MS_LOG(INFO) << "Hash table info:"
                 << " param_key:" << param_key << ", embedding table name:" << param_name
                 << ", vocab size:" << vocab_size << ", embedding size:" << embedding_size
                 << ", device cache size:" << cache_vocab_size << ", host cache size:" << host_cache_vocab_size
                 << ", device cache address:" << item.second.address.addr
                 << ", host cache address:" << item.second.host_address.get();
  }
}

EmbeddingStorageManager &EmbeddingStorageManager::GetInstance() {
  static EmbeddingStorageManager instance{};
  return instance;
}

namespace {
/**
 * @brief Create a new embedding storage instance for specific key and value type, and add the instance to
 * EmbeddingStorageManager, this function is the implementation of function `CreateEmbeddingStorage`.
 * @param[in] `embedding_key`: The unique parameter key for embedding table.
 * @param[in] `embedding_dim`: The length of each embedding vector.
 * @param[in] `capacity`: The capacity for new embedding storage.
 */
template <typename KeyType, typename ValueType>
void CreateEmbeddingStorageFunc(int32_t embedding_key, size_t embedding_dim, size_t capacity) {
  std::shared_ptr<storage::AbstractEmbeddingStorage> embedding_storage = nullptr;
  if (!EmbeddingCacheTableManager::GetInstance().is_sparse_format()) {
    embedding_storage =
      std::make_shared<storage::DenseEmbeddingStorage<KeyType, ValueType>>(embedding_key, embedding_dim, capacity);
  } else {
    embedding_storage =
      std::make_shared<storage::SparseEmbeddingStorage<KeyType, ValueType>>(embedding_key, embedding_dim, capacity);
  }
  MS_EXCEPTION_IF_NULL(embedding_storage);
  EmbeddingStorageManager::GetInstance().Add(embedding_key, embedding_storage);
}

// Key-Value type pair -> CreateEmbeddingStorageFunc map.
const std::map<std::pair<TypeId, TypeId>, std::function<void(int32_t, size_t, size_t)>> kCreateEmbeddingStorageFuncs = {
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeBool), CreateEmbeddingStorageFunc<int32_t, bool>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt8), CreateEmbeddingStorageFunc<int32_t, int8_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt16), CreateEmbeddingStorageFunc<int32_t, int16_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt32), CreateEmbeddingStorageFunc<int32_t, int32_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt64), CreateEmbeddingStorageFunc<int32_t, int64_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeUInt8), CreateEmbeddingStorageFunc<int32_t, uint8_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeUInt16), CreateEmbeddingStorageFunc<int32_t, uint16_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeUInt32), CreateEmbeddingStorageFunc<int32_t, uint32_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeUInt64), CreateEmbeddingStorageFunc<int32_t, uint64_t>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeFloat16), CreateEmbeddingStorageFunc<int32_t, float16>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeFloat32), CreateEmbeddingStorageFunc<int32_t, float>},
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeFloat64), CreateEmbeddingStorageFunc<int32_t, double>},

  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeBool), CreateEmbeddingStorageFunc<int64_t, bool>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeInt8), CreateEmbeddingStorageFunc<int64_t, int8_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeInt16), CreateEmbeddingStorageFunc<int64_t, int16_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeInt32), CreateEmbeddingStorageFunc<int64_t, int32_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeInt64), CreateEmbeddingStorageFunc<int64_t, int64_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeUInt8), CreateEmbeddingStorageFunc<int64_t, uint8_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeUInt16), CreateEmbeddingStorageFunc<int64_t, uint16_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeUInt32), CreateEmbeddingStorageFunc<int64_t, uint32_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeUInt64), CreateEmbeddingStorageFunc<int64_t, uint64_t>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeFloat16), CreateEmbeddingStorageFunc<int64_t, float16>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeFloat32), CreateEmbeddingStorageFunc<int64_t, float>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeFloat64), CreateEmbeddingStorageFunc<int64_t, double>}};
}  // namespace

void CreateEmbeddingStorage(std::pair<TypeId, TypeId> key_value_types, int32_t embedding_key, size_t embedding_dim,
                            size_t capacity) {
  const auto &iter = kCreateEmbeddingStorageFuncs.find(key_value_types);
  if (iter == kCreateEmbeddingStorageFuncs.end()) {
    MS_LOG(EXCEPTION) << "Can not find function to create embedding storage for key type:"
                      << TypeIdToString(key_value_types.first)
                      << ", value type:" << TypeIdToString(key_value_types.second);
  }
  iter->second(embedding_key, embedding_dim, capacity);
}

EmbeddingDeviceCache::EmbeddingDeviceCache(size_t batch_ids_num, size_t cache_vocab_size)
    : hash_swap_index_addr_(nullptr), hash_swap_value_addr_(nullptr) {
  device_to_host_index = std::make_unique<int[]>(batch_ids_num);
  device_to_host_ids = std::make_unique<int[]>(batch_ids_num);
  host_to_device_index = std::make_unique<int[]>(batch_ids_num);
  host_to_device_ids = std::make_unique<int[]>(batch_ids_num);
  device_hash_map_ = std::make_shared<EmbeddingHashMap>(0, cache_vocab_size);
}

EmbeddingHostCache::EmbeddingHostCache(size_t batch_ids_num, size_t host_cache_vocab_size) {
  host_to_server_index = std::make_unique<int[]>(batch_ids_num);
  host_to_server_ids = std::make_unique<int[]>(batch_ids_num);
  server_to_host_index = std::make_unique<int[]>(batch_ids_num);
  server_to_host_ids = std::make_unique<int[]>(batch_ids_num);
  new_id_index = std::make_unique<int[]>(batch_ids_num);
  host_to_device_index = std::make_unique<int[]>(batch_ids_num);
  device_to_host_index = std::make_unique<int[]>(batch_ids_num);
  host_hash_map_ = std::make_shared<EmbeddingHashMap>(0, host_cache_vocab_size);
}

void EmbeddingStorageManager::Add(int32_t param_key,
                                  const std::shared_ptr<storage::AbstractEmbeddingStorage> &embed_storage) {
  MS_EXCEPTION_IF_NULL(embed_storage);
  embedding_storages_[param_key] = embed_storage;
}

std::shared_ptr<storage::AbstractEmbeddingStorage> EmbeddingStorageManager::Get(int32_t param_key) {
  const auto &iter = embedding_storages_.find(param_key);
  if (iter != embedding_storages_.end()) {
    return iter->second;
  }
  MS_LOG(EXCEPTION) << "Can not find embedding storage for parameter key[" << param_key << "].";
}

void EmbeddingStorageManager::Clear() {
  for (const auto &item : embedding_storages_) {
    const auto &embedding_storage = item.second;
    MS_EXCEPTION_IF_NULL(embedding_storage);
    embedding_storage->Finalize();
  }

  embedding_storages_.clear();
}
}  // namespace distributed
}  // namespace mindspore
