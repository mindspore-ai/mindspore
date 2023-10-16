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
#include <thread>
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

void EmbeddingCacheTableManager::Initialize() {
  auto worker_num = ps::PSContext::instance()->worker_num();
  multi_batch_threshold_ = worker_num > 1 ? 1 : kMultiBatchThreshold;
  GetEmbeddingTableSliceBound();

  device::DeviceContextKey host_key = {"CPU", 0};
  cpu_device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(cpu_device_context_);
  cpu_device_context_->Initialize();
}

void EmbeddingCacheTableManager::Finalize(const device::DeviceContext *device_context) {
  hash_tables_.clear();

  device_hash_map_ = nullptr;
  host_hash_map_ = nullptr;

  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  if (hash_swap_index_addr_) {
    device_context->device_res_manager_->FreeMemory(hash_swap_index_addr_);
  }
  if (hash_swap_value_addr_) {
    device_context->device_res_manager_->FreeMemory(hash_swap_value_addr_);
  }
  for (auto &item : hash_tables_) {
    if (item.second.host_address) {
      MS_EXCEPTION_IF_NULL(cpu_device_context_);
      MS_EXCEPTION_IF_NULL(cpu_device_context_->device_res_manager_);
      cpu_device_context_->device_res_manager_->FreeMemory(item.second.host_address);
    }
  }
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
      MS_EXCEPTION_IF_CHECK_FAIL(device_context->device_res_manager_->AllocateMemory(device_address),
                                 "Allocate device memory for embedding table failed.");
    }
    item.second.address = Address(device_address->GetMutablePtr(), device_address->GetSize());

    size_t embedding_size = item.second.embedding_size;
    auto &host_address = item.second.host_address;
    host_address = reinterpret_cast<float *>(
      cpu_device_context_->device_res_manager_->AllocateMemory(host_cache_size_ * embedding_size * sizeof(float)));
    MS_EXCEPTION_IF_NULL(host_address);

    max_embedding_size = (embedding_size > max_embedding_size) ? embedding_size : max_embedding_size;
  }

  device_hash_map_ = std::make_shared<EmbeddingHashMap>(device_cache_size_);
  MS_EXCEPTION_IF_NULL(device_hash_map_);
  host_hash_map_ = std::make_shared<EmbeddingHashMap>(host_cache_size_);
  MS_EXCEPTION_IF_NULL(host_hash_map_);

  hash_swap_index_addr_ = reinterpret_cast<int *>(
    device_context->device_res_manager_->AllocateMemory(batch_ids_num_ * sizeof(int) * multi_batch_threshold_));
  MS_EXCEPTION_IF_NULL(hash_swap_index_addr_);
  hash_swap_value_addr_ = reinterpret_cast<float *>(device_context->device_res_manager_->AllocateMemory(
    max_embedding_size * batch_ids_num_ * sizeof(float) * multi_batch_threshold_));
  MS_EXCEPTION_IF_NULL(hash_swap_value_addr_);
}

void EmbeddingCacheTableManager::GetEmbeddingTableSliceBound() {
  auto worker_num = ps::PSContext::instance()->worker_num();
  auto server_num = ps::PSContext::instance()->server_num();
  if (worker_num == 0) {
    return;
  }
  if (is_sparse_format() && (worker_num > 1 || server_num > 1)) {
    MS_LOG(EXCEPTION) << "The sparse format can not support multi worker or multi server currently.";
  }

  uint32_t rank_id = 0;
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  auto node = distributed::cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);
  rank_id = node->rank_id();
#endif

  if (!is_sparse_format()) {
    int local_shard_size = UintToInt(SizeToUint(vocab_size_) / worker_num);
    if (vocab_size_ % worker_num != 0) {
      local_shard_size += 1;
    }
    local_embedding_slice_bounds_.first = local_shard_size * UintToInt(rank_id);
    local_embedding_slice_bounds_.second =
      std::min(local_embedding_slice_bounds_.first + local_shard_size, SizeToInt(vocab_size_));
  } else {
    local_embedding_slice_bounds_.first = 0;
    local_embedding_slice_bounds_.second = INT_MAX;
  }
  local_device_cache_bounds_.first = SizeToInt(device_cache_size_) * UintToInt(rank_id);
  local_device_cache_bounds_.second = local_device_cache_bounds_.first + SizeToInt(device_cache_size_);
  MS_LOG(INFO) << "Worker num:" << worker_num << ", rank id:" << rank_id
               << ", id begin:" << local_embedding_slice_bounds_.first
               << ", id end:" << local_embedding_slice_bounds_.second
               << ", cache indices begin: " << local_device_cache_bounds_.first
               << ", cache indices end: " << local_device_cache_bounds_.second << ", vocab_size: " << vocab_size_
               << ", device cache size: " << device_cache_size_;
}

int EmbeddingCacheTableManager::cache_indices_lower_bound() const { return local_device_cache_bounds_.first; }

void EmbeddingCacheTableManager::DumpHashTables() const {
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
                 << ", host cache address:" << item.second.host_address;
  }
}

bool EmbeddingCacheTableManager::enable_pipeline() const {
  return ps::PSContext::instance()->is_worker() && ps::PSContext::instance()->cache_enable();
}

int32_t EmbeddingCacheTableManager::StoreWarmUpPtr(const int32_t param_key, const tensor::TensorPtr &tensor_ptr) {
  return StoreWarmUpPtr(param_key, nullptr, tensor_ptr, nullptr);
}

int32_t EmbeddingCacheTableManager::StoreWarmUpPtr(const int32_t param_key, const tensor::TensorPtr &key_ptr,
                                                   const tensor::TensorPtr &value_ptr,
                                                   const tensor::TensorPtr &status_ptr) {
  MS_LOG(INFO) << "Enter store warm up ptr, param_key : " << param_key << ".";
  MS_EXCEPTION_IF_NULL(value_ptr);
  std::unique_lock<std::mutex> lock(host_cache_mutex_);
  auto ret = host_cache_ptrs_.find(param_key);
  if (ret != host_cache_ptrs_.end()) {
    MS_LOG(WARNING) << "Store warm up ptr duplicate, id : " << param_key << ".";
  }
  (void)host_cache_ptrs_.try_emplace(param_key, key_ptr, value_ptr, status_ptr);
  MS_LOG(INFO) << "Exit store warm up ptr, host cache ptrs size : " << host_cache_ptrs_.size()
               << ", hash tables size : " << hash_tables_.size() << ".";
  return 0;
}

const HashTableInfo *EmbeddingCacheTableManager::FindHashTablesByParamKey(const int param_key) {
  const auto &iter = std::find_if(hash_tables_.begin(), hash_tables_.end(),
                                  [this, param_key](const auto &data) { return data.second.param_key_ == param_key; });
  return iter != hash_tables_.end() ? &(iter->second) : nullptr;
}

void EmbeddingCacheTableManager::WarmUpHostCacheSync(const int32_t batch_count) {
  MS_LOG(INFO) << "Enter warm up host cache sync, batch_count : " << batch_count << ".";
  auto cache_ptr_size = host_cache_ptrs_.size();
  auto hash_table_size = hash_tables_.size();
  if (cache_ptr_size != hash_table_size) {
    MS_LOG(WARNING) << "Host cache ptrs size : " << cache_ptr_size
                    << " is not equal to hash table size : " << hash_table_size
                    << ", will skip warm up host cache sync.";
    std::unique_lock<std::mutex> lock(host_cache_mutex_);
    host_cache_promise_->set_value(false);
    lock.unlock();
    host_cache_ptrs_.clear();
    return;
  }

  for (auto &item : host_cache_ptrs_) {
    WarmUpHostCacheItemBatch(batch_count, item);
  }
  std::unique_lock<std::mutex> lock(host_cache_mutex_);
  host_cache_promise_->set_value(true);
  lock.unlock();
  host_cache_ptrs_.clear();
  MS_LOG(INFO) << "Exit warm up host cache sync.";
}

void EmbeddingCacheTableManager::WarmUpHostCacheAsync(const int32_t batch_count) {
  MS_LOG(DEBUG) << "Enter warm up host cache async, batch_count : " << batch_count << ".";
  std::unique_lock<std::mutex> lock(host_cache_mutex_);
  if (host_cache_promise_ != nullptr) {
    lock.unlock();
    MS_LOG(WARNING) << "Host cache promise is not null, cache sync has already done.";
    return;
  }
  host_cache_promise_ = std::make_shared<std::promise<bool>>();
  lock.unlock();
  std::thread([this, batch_count]() { WarmUpHostCacheSync(batch_count); }).detach();
  MS_LOG(DEBUG) << "Exit warm up host cache async.";
}

std::pair<std::shared_ptr<std::future<bool>>, bool> EmbeddingCacheTableManager::GetWarmUpHostCacheAsyncStatus() {
  MS_LOG(DEBUG) << "Enter get warm up host cache async status.";
  std::unique_lock<std::mutex> lock(host_cache_mutex_);
  if (host_cache_promise_ == nullptr) {
    return std::make_pair(nullptr, false);
  }
  return std::make_pair(std::make_shared<std::future<bool>>(host_cache_promise_->get_future()), true);
}

bool EmbeddingCacheTableManager::WaitForWarmUpHostCacheComplete() {
  MS_LOG(DEBUG) << "Enter wait for warm up host cache complete.";
  const int32_t batch_count = 4;
  WarmUpHostCacheAsync(batch_count);
  const auto &[complete_future, status] = GetWarmUpHostCacheAsyncStatus();
  return status ? complete_future->get() : status;
}

tensor::TensorPtr generate_key_tensor_ptr(const tensor::TensorPtr &tensor_ptr) {
  auto &vec = tensor_ptr->shape();
  auto cel_num = static_cast<int>(vec[0]);
  std::vector<int32_t> key_vec(cel_num);
  for (auto i = 0; i != cel_num; i++) {
    key_vec[i] = i;
  }
  return std::make_shared<tensor::Tensor>(key_vec);
}

void EmbeddingCacheTableManager::WarmUpHostCacheItemBatch(const int32_t batch_count, const WarmUpCacheMapEntry &entry) {
  MS_LOG(DEBUG) << "Enter warm up host cache item batch.";
  auto key_ptr = std::get<0>(entry.second);
  auto value_ptr = std::get<1>(entry.second);
  MS_EXCEPTION_IF_NULL(value_ptr);
  // Key tensor may be nullptr since we stored single value tensor.
  if (key_ptr == nullptr) {
    MS_LOG(INFO) << "key_ptr is nullptr, generate key tensor.";
    key_ptr = generate_key_tensor_ptr(value_ptr);
  }
  auto &vec = key_ptr->shape();
  auto l_len = static_cast<int>(vec[0]);
  const int32_t default_batch_count = 1;
  const int validate_batch_count = batch_count < default_batch_count ? default_batch_count : batch_count;
  int batch_size = l_len / validate_batch_count;
  if (l_len % validate_batch_count != 0) {
    batch_size++;
  }

  if (host_hash_map_ == nullptr) {
    MS_LOG(WARNING) << "Embedding hash map of embedding host cache is nullptr, will skip warm up.";
    return;
  }

  auto hash_table_info_ptr = FindHashTablesByParamKey(entry.first);
  if (hash_table_info_ptr == nullptr) {
    MS_LOG(WARNING) << "Hash table info is nullptr, will skip warm up.";
    return;
  }

  size_t host_length = (hash_table_info_ptr->host_cache_vocab_size * hash_table_info_ptr->embedding_size) << 2;
  auto &value_shape = value_ptr->shape();
  size_t value_len = 0;
  (void)std::for_each(value_shape.begin() + 1, value_shape.end(), [&](int n) { value_len += n; });
  MS_EXCEPTION_IF_NULL(value_ptr->data_ptr());
  value_len *= static_cast<size_t>(value_ptr->data_ptr()->itemsize());
  size_t value_expected_len = value_len * (value_shape[0] + 1);
  MS_EXCEPTION_IF_CHECK_FAIL(value_expected_len <= host_length, "Size of value tensor is overflow.");

  for (int i = 0; i < l_len; i += batch_size) {
    WarmUpHostCacheItem(host_hash_map_, hash_table_info_ptr, entry, i, std::min(i + batch_size, l_len), value_len);
  }
  MS_LOG(DEBUG) << "Exit warm up host cache item batch.";
}

void EmbeddingCacheTableManager::WarmUpHostCacheItem(const std::shared_ptr<EmbeddingHashMap> &embedding_hash_map,
                                                     const HashTableInfo *hash_table_info_ptr,
                                                     const WarmUpCacheMapEntry &entry, const int start, const int end,
                                                     const size_t value_len) {
  // Value type is float, bit num is 2
  const int shift_bit_num = 2;
  if (hash_table_info_ptr->embedding_size != (value_len >> shift_bit_num)) {
    MS_LOG(WARNING) << "Hash table info embedding_size : " << hash_table_info_ptr->embedding_size
                    << " is not equal to value_len : " << value_len << ".";
    return;
  }

  auto key_ptr = std::get<0>(entry.second);
  auto key_data_ptr = key_ptr->data_ptr();
  for (ssize_t i = start; i != end; i++) {
    auto key_data_type = key_ptr->data_type();
    int64_t key = 0;
    switch (key_data_type) {
      case TypeId::kNumberTypeInt32:
      case TypeId::kNumberTypeUInt32: {
        auto int_ptr = static_cast<int *>(key_ptr->data_c());
        key = *(int_ptr + i);
      } break;
      case TypeId::kNumberTypeInt64:
      case TypeId::kNumberTypeUInt64: {
        auto int64_ptr = static_cast<int64_t *>(key_ptr->data_c());
        key = *(int64_ptr + i);
      } break;
      default:
        MS_LOG(WARNING) << "Invalid key_data_type : " << key_data_type << ".";
        return;
    }

    int id = embedding_hash_map->GetOrInsertDataUnsafe(static_cast<int>(key));
    if (id == kInvalidIndexValue) {
      MS_LOG(WARNING) << "Embedding hash map is full, exit warm up process.";
      break;
    }

    size_t offset = static_cast<size_t>(id) * value_len;
    auto host_address = hash_table_info_ptr->host_address;
    auto des_ptr = AddressOffset(host_address, 0);
    auto value_data_ptr = std::get<1>(entry.second)->data_c();
    auto src_ptr = AddressOffset(value_data_ptr, 0);
    auto ret_code = memcpy_s(des_ptr + offset, value_len, src_ptr + i * value_len, value_len);
    if (ret_code != EOK) {
      MS_LOG(EXCEPTION) << "Failed to copy data, memcpy_s errorno: " << ret_code;
    }
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
  {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeBFloat16),
   CreateEmbeddingStorageFunc<int32_t, bfloat16>},

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
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeFloat64), CreateEmbeddingStorageFunc<int64_t, double>},
  {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeBFloat16),
   CreateEmbeddingStorageFunc<int64_t, bfloat16>}};
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

EmbeddingDeviceCache::EmbeddingDeviceCache(size_t batch_ids_num) {
  device_to_host_index = std::make_unique<int[]>(batch_ids_num);
  device_to_host_ids = std::make_unique<int[]>(batch_ids_num);
  host_to_device_index = std::make_unique<int[]>(batch_ids_num);
  host_to_device_ids = std::make_unique<int[]>(batch_ids_num);
}

EmbeddingHostCache::EmbeddingHostCache(size_t batch_ids_num) {
  host_to_server_index = std::make_unique<int[]>(batch_ids_num);
  host_to_server_ids = std::make_unique<int[]>(batch_ids_num);
  server_to_host_index = std::make_unique<int[]>(batch_ids_num);
  server_to_host_ids = std::make_unique<int[]>(batch_ids_num);
  new_id_index = std::make_unique<int[]>(batch_ids_num);
  host_to_device_index = std::make_unique<int[]>(batch_ids_num);
  device_to_host_index = std::make_unique<int[]>(batch_ids_num);
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
