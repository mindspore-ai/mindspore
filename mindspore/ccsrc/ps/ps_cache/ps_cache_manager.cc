/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include "ps/ps_cache/ps_cache_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

using mindspore::kernel::Address;
namespace mindspore {
namespace ps {
void PsCacheManager::InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                                         size_t vocab_size) {
  if (cache_vocab_size == 0 || embedding_size == 0 || vocab_size == 0) {
    MS_LOG(EXCEPTION) << "The size of hash table can not equal to zero.";
  }
  hash_tables_[param_name].cache_vocab_size = cache_vocab_size;
  hash_tables_[param_name].host_cache_vocab_size = cache_vocab_size * kHostCacheScaleFactor;
  hash_tables_[param_name].embedding_size = embedding_size;
  hash_tables_[param_name].vocab_size = vocab_size;

  if (vocab_size_ == 0) {
    vocab_size_ = vocab_size;
  }
  if (cache_vocab_size_ == 0) {
    cache_vocab_size_ = cache_vocab_size;
  }
  if (host_cache_vocab_size_ == 0) {
    host_cache_vocab_size_ = cache_vocab_size * kHostCacheScaleFactor;
  }
}

void PsCacheManager::ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name,
                                           size_t cache_vocab_size, size_t embedding_size) {
  if (cache_vocab_size == 0 || embedding_size == 0) {
    MS_LOG(EXCEPTION) << "The size of hash table can not equal to zero.";
  }
  if (new_param_name.empty() || cur_param_name.empty()) {
    MS_LOG(EXCEPTION) << "Parameter name can not be empty.";
  }
  if (new_param_name == cur_param_name) {
    return;
  }
  auto iter = hash_tables_.find(cur_param_name);
  if (iter != hash_tables_.end()) {
    hash_tables_.emplace(new_param_name, iter->second);
    hash_tables_.erase(iter);
  } else {
    hash_tables_[new_param_name].cache_vocab_size = cache_vocab_size;
    hash_tables_[new_param_name].embedding_size = embedding_size;
  }
}

void PsCacheManager::InsertWeightInitInfo(const std::string &param_name, size_t global_seed, size_t op_seed) {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find parameter[" << param_name << "] in hash table.";
  }
  auto &hash_table_info = iter->second;
  hash_table_info.param_init_info_.param_type_ = kWeight;
  hash_table_info.param_init_info_.global_seed_ = global_seed;
  hash_table_info.param_init_info_.op_seed_ = op_seed;
}

void PsCacheManager::InsertAccumuInitInfo(const std::string &param_name, float init_val) {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find parameter[" << param_name << "] in hash table.";
  }
  auto &hash_table_info = iter->second;
  hash_table_info.param_init_info_.param_type_ = kAccumulation;
  hash_table_info.param_init_info_.init_val_ = init_val;
}

void PsCacheManager::CloneHashTable(const std::string &dest_param_name, const std::string &src_param_name) {
  if (dest_param_name == src_param_name) {
    MS_LOG(INFO) << "The dest_param_name is same as src_param_name";
    return;
  }
  auto iter = hash_tables_.find(src_param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "The source hash table[" << src_param_name << "] does not exist, clone failed.";
  }
  hash_tables_.emplace(dest_param_name, iter->second);
}

const Address &PsCacheManager::QueryHashTableAddr(const std::string &param_name) const {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find device_address of " << param_name;
  }
  return iter->second.device_address;
}

void PsCacheManager::Initialize() {
  MS_LOG(INFO) << "PS cache initialize.";
  if (!worker.running()) {
    Util::SetInternalEnvVar();
    worker.Run();
  }
  embedding_device_cache_ = std::make_shared<EmbeddingDeviceCache>(batch_elements_, cache_vocab_size_);
  embedding_host_cache_ = std::make_shared<EmbeddingHostCache>(batch_elements_, host_cache_vocab_size_);
  InitParameterServer();
  AllocMemForHashTable();
  SetLocalIdRank();
  initialized_ps_cache_ = true;
}

void PsCacheManager::InitParameterServer() {
  for (const auto &item : hash_tables_) {
    const auto &param_name = item.first;
    size_t key = worker.SetParamKey(param_name);
    size_t row_count = item.second.vocab_size;
    std::vector<size_t> keys{key, key, key, key};
    std::vector<float> values{
      SizeToFloat(item.second.vocab_size), SizeToFloat(item.second.embedding_size), 1, 1, 1, 1, 1};
    std::vector<int64_t> lens{2, 2, 3};
    const auto &hash_table_info = item.second;
    const auto &param_init_info = hash_table_info.param_init_info_;
    if (param_init_info.param_type_ == kWeight) {
      lens.push_back(0);
      values.push_back(SizeToFloat(param_init_info.global_seed_));
      values.push_back(SizeToFloat(param_init_info.op_seed_));
    } else if (param_init_info.param_type_ == kAccumulation) {
      lens.push_back(1);
      values.push_back(param_init_info.init_val_);
    }
    // if worker role
    worker.AddEmbeddingTable(key, row_count);
    worker.InitPSEmbeddingTable(keys, values, lens);
  }
}

void PsCacheManager::AllocMemForHashTable() {
  MS_EXCEPTION_IF_NULL(embedding_device_cache_);
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->cache_);
  size_t max_embedding_size = 0;
  for (auto &item : hash_tables_) {
    size_t embedding_size = item.second.embedding_size;
    auto &device_address = item.second.device_address;
    device_address.size = cache_vocab_size_ * embedding_size * sizeof(float);
    auto addr = embedding_device_cache_->cache_->MallocMemory(device_address.size);
    MS_EXCEPTION_IF_NULL(addr);
    device_address.addr = addr;

    auto &host_address = item.second.host_address;
    auto host_address_ptr = new int[host_cache_vocab_size_ * embedding_size];
    MS_EXCEPTION_IF_NULL(host_address_ptr);
    host_address = std::shared_ptr<int[]>(host_address_ptr, std::default_delete<int[]>());
    MS_EXCEPTION_IF_NULL(host_address);

    max_embedding_size = (embedding_size > max_embedding_size) ? embedding_size : max_embedding_size;
  }
  embedding_device_cache_->hash_swap_index_addr_ =
    reinterpret_cast<int *>(embedding_device_cache_->cache_->MallocMemory(batch_elements_ * sizeof(int)));
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->hash_swap_index_addr_);
  embedding_device_cache_->hash_swap_value_addr_ = reinterpret_cast<float *>(
    embedding_device_cache_->cache_->MallocMemory(max_embedding_size * batch_elements_ * sizeof(float)));
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->hash_swap_value_addr_);
  embedding_device_cache_->cache_->MallocConstantMemory(cache_vocab_size_);
}

void PsCacheManager::SetLocalIdRank() {
  auto worker_num = ::ps::NumWorkers();
  auto worker_id = ::ps::MyRank();
  auto local_shard_size = FloatToSize(std::ceil(SizeToFloat(vocab_size_) / worker_num));
  range_bound_.first = local_shard_size * worker_id;
  range_bound_.second = std::min(range_bound_.first + local_shard_size, vocab_size_);
  MS_LOG(INFO) << "Worker num:" << worker_num << ", worker id:" << worker_id << ", rank id begin:" << range_bound_.first
               << ", rank id end:" << range_bound_.second;
}

std::string PsCacheManager::channel_name() {
  std::lock_guard<std::mutex> locker(channel_mutex_);
  return channel_name_;
}

void PsCacheManager::set_channel_name(const std::string channel_name) {
  if (channel_name_ == channel_name) {
    return;
  }
  std::lock_guard<std::mutex> locker(channel_mutex_);
  channel_name_ = channel_name;
}

void PsCacheManager::IncreaseStep() {
  if (data_step_ >= UINT64_MAX) {
    MS_LOG(EXCEPTION) << "The data step (" << data_step_ << ")  <<  will exceed the maximum value of uint64_t.";
  }
  data_step_++;
  set_current_graph_step();
}

void PsCacheManager::IncreaseGraphStep(const std::string &channel_name) {
  if (graph_step_ >= UINT64_MAX) {
    MS_LOG(EXCEPTION) << "The graph step(" << graph_step_ << ")  <<  will exceed the maximum value of uint64_t.";
  }
  graph_step_++;
  set_channel_name(channel_name);
  PsDataPrefetch::GetInstance().TryWakeChannel(channel_name);
  data_prase_.notify_one();
}
}  // namespace ps
}  // namespace mindspore
