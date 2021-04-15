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
  if (vocab_cache_size_ == 0) {
    vocab_cache_size_ = cache_vocab_size;
  }
  if (host_vocab_cache_size_ == 0) {
    host_vocab_cache_size_ = cache_vocab_size * kHostCacheScaleFactor;
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
  if (hash_table_info.param_init_info_.param_type_ != kUnKnown) {
    return;
  }
  MS_LOG(INFO) << "Insert embedding table init info:" << param_name << ", global seed:" << global_seed
               << ", op seed:" << op_seed;
  hash_table_info.param_init_info_.param_type_ = kWeight;
  hash_table_info.param_init_info_.global_seed_ = global_seed;
  hash_table_info.param_init_info_.op_seed_ = op_seed;
  if (CheckFinishInsertInitInfo()) {
    finish_insert_init_info_ = true;
    insert_init_info_.notify_one();
  }
}

void PsCacheManager::InsertAccumuInitInfo(const std::string &param_name, float init_val) {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find parameter[" << param_name << "] in hash table.";
  }
  auto &hash_table_info = iter->second;
  if (hash_table_info.param_init_info_.param_type_ != kUnKnown) {
    return;
  }
  MS_LOG(INFO) << "Insert accumulation init info:" << param_name << ", init value:" << init_val;
  hash_table_info.param_init_info_.param_type_ = kAccumulation;
  hash_table_info.param_init_info_.init_val_ = init_val;
  if (CheckFinishInsertInitInfo()) {
    finish_insert_init_info_ = true;
    insert_init_info_.notify_one();
  }
}

bool PsCacheManager::CheckFinishInsertInitInfo() const {
  for (const auto &item : hash_tables_) {
    const auto &hash_table_info = item.second;
    const auto &param_init_info = hash_table_info.param_init_info_;
    if (param_init_info.param_type_ == kUnKnown) {
      return false;
    }
  }
  MS_LOG(INFO) << "Finish inserting embedding table init info.";
  return true;
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
    MS_LOG(EXCEPTION) << "Can not find device address of " << param_name;
  }
  return iter->second.device_address;
}

const size_t &PsCacheManager::QueryHashTableSize(const std::string &param_name) const {
  auto iter = hash_tables_.find(param_name);
  if (iter == hash_tables_.end()) {
    MS_LOG(EXCEPTION) << "Can not find vocab cache size of " << param_name;
  }
  return iter->second.cache_vocab_size;
}

void PsCacheManager::Initialize() {
  MS_LOG(INFO) << "PS cache initialize.";
  if (!Worker::GetInstance().running()) {
    Worker::GetInstance().Run();
  }
  embedding_device_cache_ = std::make_shared<EmbeddingDeviceCache>(batch_elements_, vocab_cache_size_);
  embedding_host_cache_ = std::make_shared<EmbeddingHostCache>(batch_elements_, host_vocab_cache_size_);
  AddEmbeddingTable();
  AllocMemForHashTable();
  SetLocalIdRank();
  DumpHashTables();
  initialized_ps_cache_ = true;
}

void PsCacheManager::AddEmbeddingTable() const {
  for (const auto &item : hash_tables_) {
    const auto &param_name = item.first;
    size_t key = Worker::GetInstance().SetParamKey(param_name);
    size_t row_count = item.second.vocab_size;
    // if worker role
    Worker::GetInstance().AddEmbeddingTable(key, row_count);
  }
}

void PsCacheManager::InitParameterServer() {
  MS_LOG(INFO) << "PS embedding cache table init begin:" << finish_insert_init_info_;
  std::unique_lock<std::mutex> locker(data_mutex_);
  insert_init_info_.wait(locker, [this] { return finish_insert_init_info_ == true || running_ == false; });
  if (!running_) {
    return;
  }
  for (const auto &item : hash_tables_) {
    const auto &param_name = item.first;
    size_t key = Worker::GetInstance().SetParamKey(param_name);
    const auto &hash_table_info = item.second;
    const auto &param_init_info = hash_table_info.param_init_info_;

    std::vector<size_t> input_shape = {item.second.vocab_size, item.second.embedding_size};
    std::vector<size_t> indices_shape = {1, 1};
    std::vector<size_t> output_shape = {1, 1, 1};
    ParamInitInfoMessage info;
    info.set_param_type(param_init_info.param_type_);
    info.set_init_val(param_init_info.init_val_);
    info.set_global_seed(param_init_info.global_seed_);
    info.set_op_seed(param_init_info.op_seed_);
    // if worker role
    Worker::GetInstance().InitPSEmbeddingTable(key, input_shape, indices_shape, output_shape, info);
  }

  finish_init_parameter_server_ = true;
  data_prase_.notify_one();
  MS_LOG(INFO) << "PS embedding cache table init end.";
}

void PsCacheManager::InitDataChannel() {
  MS_LOG(INFO) << "PS embedding cache data channel init begin.";
  auto channel = channel_name();
  if (channel.empty()) {
    std::unique_lock<std::mutex> locker(data_mutex_);
    data_prase_.wait(locker, [this] { return !channel_name_.empty() || running_ == false; });
    if (!running_) {
      return;
    }
  }
  MS_LOG(INFO) << "PS embedding cache data channel  init end.";
}

void PsCacheManager::AllocMemForHashTable() {
  MS_EXCEPTION_IF_NULL(embedding_device_cache_);
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->cache_);
  size_t max_embedding_size = 0;
  for (auto &item : hash_tables_) {
    size_t embedding_size = item.second.embedding_size;
    auto &device_address = item.second.device_address;
    device_address.size = vocab_cache_size_ * embedding_size * sizeof(float);
    auto addr = embedding_device_cache_->cache_->MallocMemory(device_address.size);
    MS_EXCEPTION_IF_NULL(addr);
    device_address.addr = addr;

    auto &host_address = item.second.host_address;
    host_address =
      std::shared_ptr<float[]>(new float[host_vocab_cache_size_ * embedding_size], std::default_delete<float[]>());
    MS_EXCEPTION_IF_NULL(host_address);

    max_embedding_size = (embedding_size > max_embedding_size) ? embedding_size : max_embedding_size;
  }
  embedding_device_cache_->hash_swap_index_addr_ =
    reinterpret_cast<int *>(embedding_device_cache_->cache_->MallocMemory(batch_elements_ * sizeof(int)));
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->hash_swap_index_addr_);
  embedding_device_cache_->hash_swap_value_addr_ = reinterpret_cast<float *>(
    embedding_device_cache_->cache_->MallocMemory(max_embedding_size * batch_elements_ * sizeof(float)));
  MS_EXCEPTION_IF_NULL(embedding_device_cache_->hash_swap_value_addr_);
  if (!(embedding_device_cache_->cache_->MallocConstantMemory(vocab_cache_size_))) {
    MS_LOG(EXCEPTION) << "MallocConstantMemory failed.";
  }
}

void PsCacheManager::SetLocalIdRank() {
  auto worker_num = PSContext::instance()->initial_worker_num();
  auto local_shard_size = FloatToInt(std::ceil(SizeToFloat(vocab_size_) / worker_num));
  vocab_cache_size_diff_ = local_shard_size - SizeToInt(vocab_cache_size_);
  emb_table_slice_bounds_.first = local_shard_size * rank_id_;
  emb_table_slice_bounds_.second = std::min(emb_table_slice_bounds_.first + local_shard_size, SizeToInt(vocab_size_));
  cache_indices_bounds_.first = SizeToInt(vocab_cache_size_) * rank_id_;
  cache_indices_bounds_.second = cache_indices_bounds_.first + SizeToInt(vocab_cache_size_);
  MS_LOG(INFO) << "Worker num:" << worker_num << ", rank id:" << rank_id_
               << ", id begin:" << emb_table_slice_bounds_.first << ", id end:" << emb_table_slice_bounds_.second
               << ", cache indices begin: " << cache_indices_bounds_.first
               << ", cache indices end: " << cache_indices_bounds_.second
               << ", vocab_cache_size_diff: " << vocab_cache_size_diff_;
}

int PsCacheManager::cache_indices_lower_bound() const { return cache_indices_bounds_.first; }

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

bool PsCacheManager::IncreaseStep() {
  if (data_step_ >= UINT64_MAX) {
    MS_LOG(ERROR) << "The data step (" << data_step_ << ") will exceed the maximum value of uint64_t.";
    return false;
  }
  data_step_++;
  set_current_graph_step();
  if (graph_running_step_ > data_step_) {
    MS_LOG(ERROR) << "The graph running step (" << graph_running_step_ << ") exceed the data step (" << data_step_
                  << ").";
    return false;
  }
  return true;
}

void PsCacheManager::IncreaseGraphStep(const std::string &channel_name) {
  if (!running_) {
    MS_LOG(EXCEPTION) << "PS embedding cache data processing thread isn't running.";
  }
  if (graph_step_ >= UINT64_MAX) {
    MS_LOG(EXCEPTION) << "The graph step(" << graph_step_ << ") will exceed the maximum value of uint64_t.";
  }
  if (graph_step_ == 0) {
    MS_LOG(INFO) << "Graph running waiting embedding table init begin:" << finish_init_parameter_server_;
    std::unique_lock<std::mutex> locker(data_mutex_);
    data_prase_.wait(locker, [this] { return ((finish_init_parameter_server_ == true) || (running_ == false)); });
    if (!running_) {
      MS_LOG(EXCEPTION) << "PS embedding cache data processing thread isn't running.";
    }
    MS_LOG(INFO) << "Graph running waiting embedding table init end.";
  }
  graph_step_++;
  set_channel_name(channel_name);
  if (!PsDataPrefetch::GetInstance().TryWakeChannel(channel_name)) {
    MS_LOG(EXCEPTION) << "TryWakeChannel failed, channel name: " << channel_name;
  }
  data_prase_.notify_one();
}

void PsCacheManager::DoProcessData(uint32_t device_id, const void *context) {
  // PS embeddingLookup cache check.
  if (!initialized_ps_cache_) {
    MS_LOG(EXCEPTION) << "Only the sink_mode of dataset supports embeddingLookup cache in parameter server training "
                         "mode, current dataset mode is not sink_mode.";
  }
  process_data_thread_ = std::thread(&PsCacheManager::ProcessDataTask, this, device_id, context);
}

void PsCacheManager::ProcessDataTask(uint32_t device_id, const void *context) {
  MS_LOG(INFO) << "PS embedding cache process data task begin.";
  running_ = true;
  embedding_device_cache_->cache_->InitDevice(device_id, context);
  InitParameterServer();
  InitDataChannel();
  while (running_) {
    if (!ProcessData()) {
      running_ = false;
    }
  }
  MS_LOG(INFO) << "PS embedding cache process data task end.";
}

void PsCacheManager::Finalize() {
  if (running_) {
    SyncEmbeddingTable();
  }
  running_ = false;
  PsDataPrefetch::GetInstance().NotifyFinalize();
  insert_init_info_.notify_all();
  data_prase_.notify_all();
  if (process_data_thread_.joinable()) {
    process_data_thread_.join();
  }
}

bool PsCacheManager::ProcessData() {
  struct timeval start_time, end_time;
  const uint64_t kUSecondInSecond = 1000000;
  (void)gettimeofday(&start_time, nullptr);
  void *data = nullptr;
  if (!PsDataPrefetch::GetInstance().QueryData(channel_name_, &data)) {
    return false;
  }
  if (data == nullptr) {
    MS_LOG(INFO) << "No data process, channel name:" << channel_name_;
    std::unique_lock<std::mutex> locker(data_mutex_);
    (void)data_prase_.wait_for(locker, std::chrono::milliseconds(100));
    return true;
  }
  RETURN_IF_FALSE(IncreaseStep());
  auto data_size = PsDataPrefetch::GetInstance().data_size(channel_name_);
  if (data_size == 0) {
    MS_LOG(ERROR) << "The data_size can not be zero.";
    return false;
  }
  auto batch_ids = reinterpret_cast<int *>(data);
  auto batch_ids_len = data_size / sizeof(int);
  std::unique_ptr<int[]> hash_index(new int[batch_ids_len]);
  if (memset_s(&statistics_info_, sizeof(statistics_info_), 0, sizeof(statistics_info_))) {
    MS_LOG(ERROR) << "Process data memset failed.";
    return false;
  }
  // Get hash swap in/out index and ids.
  RETURN_IF_FALSE(ParseData(batch_ids, batch_ids_len, hash_index.get()));
  DumpStatisticsInfo();
  if ((device_need_wait_graph_ || host_need_wait_graph_) && (!WaitGraphRun())) {
    MS_LOG(ERROR) << "Ps cache wait graph finish failed.";
    return false;
  }
  for (const auto &item : hash_tables_) {
    auto key = Worker::GetInstance().GetParamKey(item.first);
    auto hash_info = item.second;
    RETURN_IF_FALSE(HashSwapHostToServer(key, hash_info));
    RETURN_IF_FALSE(HashSwapDeviceToHost(hash_info));
    RETURN_IF_FALSE(HashSwapServerToHost(key, hash_info));
    RETURN_IF_FALSE(HashSwapHostToDevice(hash_info));
  }
  size_t dest_len = data_size;
  // Replace the batch_ids by hash index for getNext-op getting hash index as input.
  if (memcpy_s(data, dest_len, hash_index.get(), data_size) != EOK) {
    MS_LOG(ERROR) << "Process data memcpy failed.";
    return false;
  }
  RETURN_IF_FALSE(embedding_device_cache_->cache_->SynchronizeStream());
  // Finish the data process and notify data prefetch.
  RETURN_IF_FALSE(PsDataPrefetch::GetInstance().FinalizeData(channel_name_));
  (void)gettimeofday(&end_time, nullptr);
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(DEBUG) << "Ps cache completes processing data(data step:" << data_step_
                << ",graph step:" << graph_running_step_ << " channel name:" << channel_name_
                << ", time cost:" << cost / 1000 << "ms).";
  return true;
}

bool PsCacheManager::CheckCacheHitOrOutRangeTask(const int *batch_ids, const size_t batch_ids_len, int *hash_index,
                                                 bool *in_device, bool *out_range, size_t *hash_hit_count) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(hash_hit_count);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();

  for (size_t i = 0; i < batch_ids_len; ++i) {
    if (batch_ids[i] < emb_table_slice_bounds_.first) {
      hash_index[i] = batch_ids[i] - vocab_cache_size_diff_;
      out_range[i] = true;
      continue;
    }
    if (batch_ids[i] >= emb_table_slice_bounds_.second) {
      hash_index[i] = batch_ids[i] + cache_indices_bounds_.second;
      out_range[i] = true;
      continue;
    }
    auto iter = hash_id_to_index.find(batch_ids[i]);
    if (iter != hash_id_to_index.end()) {
      hash_index[i] = iter->second + cache_indices_bounds_.first;
      if (device_hash_map->hash_step(iter->second) != data_step_) {
        ++(*hash_hit_count);
        device_hash_map->set_hash_step(iter->second, data_step_);
      }
      in_device[i] = true;
    }
  }
  return true;
}

bool PsCacheManager::CheckCacheHitOrOutRange(const int *batch_ids, const size_t batch_ids_len, int *hash_index,
                                             bool *in_device, bool *out_range) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(in_device);
  MS_ERROR_IF_NULL(out_range);

  size_t thread_num = batch_ids_len / kMinIdsPerThread + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t hash_hit_count[kMaxThreadNum] = {0};
  size_t i = 0;
  size_t task_offset = 0;

  for (; i < thread_num; ++i) {
    if (task_offset >= batch_ids_len) {
      break;
    }
    size_t task_proc_lens = batch_ids_len / thread_num + (i < (batch_ids_len % thread_num) ? 1 : 0);
    threads[i] =
      std::thread(&PsCacheManager::CheckCacheHitOrOutRangeTask, this, batch_ids + task_offset, task_proc_lens,
                  hash_index + task_offset, in_device + task_offset, out_range + task_offset, hash_hit_count + i);
    task_offset += task_proc_lens;
  }
  if (task_offset != batch_ids_len) {
    MS_LOG(WARNING) << "Ps cache check id in device inadequate, total:" << batch_ids_len << " checked:" << task_offset;
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  for (size_t j = 0; j < i; j++) {
    statistics_info_.hash_hit_count_ += hash_hit_count[j];
  }
  return true;
}

bool PsCacheManager::ResetEmbeddingHashMap() {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  const auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  MS_ERROR_IF_NULL(embedding_host_cache_);
  const auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);
  device_hash_map->Reset();
  host_hash_map->Reset();
  device_need_wait_graph_ = false;
  host_need_wait_graph_ = false;
  return true;
}

bool PsCacheManager::ParseData(const int *batch_ids, const size_t batch_ids_len, int *hash_index) {
  MS_ERROR_IF_NULL(batch_ids);
  MS_ERROR_IF_NULL(hash_index);
  statistics_info_.batch_id_count_ = batch_ids_len;
  std::unique_ptr<bool[]> in_device(new bool[batch_ids_len]);
  std::unique_ptr<bool[]> out_range(new bool[batch_ids_len]);
  if (memset_s(in_device.get(), batch_ids_len * sizeof(bool), 0, batch_ids_len * sizeof(bool))) {
    MS_LOG(EXCEPTION) << "Initialize in_device array failed.";
  }
  if (memset_s(out_range.get(), batch_ids_len * sizeof(bool), 0, batch_ids_len * sizeof(bool))) {
    MS_LOG(EXCEPTION) << "Initialize out_range array failed.";
  }
  RETURN_IF_FALSE(CheckCacheHitOrOutRange(batch_ids, batch_ids_len, hash_index, in_device.get(), out_range.get()));
  RETURN_IF_FALSE(ResetEmbeddingHashMap());
  for (size_t i = 0; i < batch_ids_len; i++) {
    if (in_device[i] || out_range[i]) {
      continue;
    }
    bool need_swap_host_to_device = true;
    bool need_swap_device_to_host = true;
    int index = INVALID_INDEX_VALUE;
    RETURN_IF_FALSE(ParseDeviceData(batch_ids[i], &need_swap_device_to_host, &need_swap_host_to_device, &index));
    hash_index[i] = index + cache_indices_bounds_.first;
    if (need_swap_host_to_device) {
      RETURN_IF_FALSE(ParseHostDataHostToDevice(batch_ids[i]));
    }
    if (need_swap_device_to_host) {
      RETURN_IF_FALSE(ParseHostDataDeviceToHost());
    }
  }
  return true;
}

bool PsCacheManager::WaitGraphRun() {
  MS_LOG(INFO) << "Hash table has no space to insert new data and retries within 2 minutes.";
  std::unique_lock<std::mutex> locker(data_mutex_);
  if (!data_prase_.wait_for(locker, std::chrono::seconds(120), [this] { return graph_step_ > graph_running_step_; })) {
    MS_LOG(ERROR) << "Ps cache data parse timeout, suggest to enlarge the cache size(graph step:" << graph_step_
                  << ", graph running step:" << graph_running_step_ << ").";
    return false;
  }
  set_current_graph_step();
  return true;
}

bool PsCacheManager::ParseDeviceData(size_t id, bool *need_swap_device_to_host, bool *need_swap_host_to_device,
                                     int *hash_index) {
  MS_ERROR_IF_NULL(need_swap_device_to_host);
  MS_ERROR_IF_NULL(need_swap_host_to_device);
  MS_ERROR_IF_NULL(hash_index);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);

  int index = INVALID_INDEX_VALUE;
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(id);
  if (iter != hash_id_to_index.end()) {
    *need_swap_device_to_host = false;
    *need_swap_host_to_device = false;
    index = iter->second;
    if (device_hash_map->hash_step(index) != data_step_) {
      statistics_info_.hash_hit_count_++;
      device_hash_map->set_hash_step(index, data_step_);
    }
  } else {
    int *device_to_host_index = embedding_device_cache_->device_to_host_index.get();
    int *device_to_host_ids = embedding_device_cache_->device_to_host_ids.get();
    int *host_to_device_index = embedding_device_cache_->host_to_device_index.get();
    int *host_to_device_ids = embedding_device_cache_->host_to_device_ids.get();
    MS_ERROR_IF_NULL(host_to_device_index);
    MS_ERROR_IF_NULL(host_to_device_ids);
    auto tmp_device_to_host_size = statistics_info_.device_to_host_size_;
    while (true) {
      index = device_hash_map->ParseData(id, device_to_host_index, device_to_host_ids, data_step_, graph_running_step_,
                                         &(statistics_info_.device_to_host_size_), &device_need_wait_graph_);
      if (index == INVALID_INDEX_VALUE) {
        if (!WaitGraphRun()) {
          return false;
        }
        continue;
      }
      host_to_device_index[statistics_info_.host_to_device_size_] = index;
      host_to_device_ids[statistics_info_.host_to_device_size_] = id;
      statistics_info_.host_to_device_size_++;
      *need_swap_device_to_host = statistics_info_.device_to_host_size_ > tmp_device_to_host_size;
      break;
    }
  }
  *hash_index = index;
  return true;
}

bool PsCacheManager::ParseHostDataHostToDevice(size_t id) {
  MS_ERROR_IF_NULL(embedding_host_cache_);
  int *host_to_device_index = embedding_host_cache_->host_to_device_index.get();
  MS_ERROR_IF_NULL(host_to_device_index);
  auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);

  const auto &hash_id_to_index = host_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(id);
  if (iter != hash_id_to_index.end()) {
    auto index = iter->second;
    if (host_hash_map->hash_step(index) != data_step_) {
      host_hash_map->set_hash_step(index, data_step_);
    }
    host_to_device_index[statistics_info_.host_to_device_size_ - 1] = index;
  } else {
    int *host_to_server_index = embedding_host_cache_->host_to_server_index.get();
    int *host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
    int *server_to_host_index = embedding_host_cache_->server_to_host_index.get();
    int *server_to_host_ids = embedding_host_cache_->server_to_host_ids.get();
    MS_ERROR_IF_NULL(server_to_host_index);
    MS_ERROR_IF_NULL(server_to_host_ids);
    while (true) {
      auto index =
        host_hash_map->ParseData(id, host_to_server_index, host_to_server_ids, data_step_, graph_running_step_,
                                 &statistics_info_.host_to_server_size_, &host_need_wait_graph_);
      if (index == INVALID_INDEX_VALUE) {
        RETURN_IF_FALSE(WaitGraphRun());
        continue;
      }
      host_to_device_index[statistics_info_.host_to_device_size_ - 1] = index;
      server_to_host_index[statistics_info_.server_to_host_size_] = index;
      server_to_host_ids[statistics_info_.server_to_host_size_++] = id;
      break;
    }
  }
  return true;
}

bool PsCacheManager::ParseHostDataDeviceToHost() {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  int *device_to_host_ids = embedding_device_cache_->device_to_host_ids.get();
  int *device_to_host_index = embedding_host_cache_->device_to_host_index.get();
  MS_ERROR_IF_NULL(device_to_host_ids);
  MS_ERROR_IF_NULL(device_to_host_index);

  auto &host_hash_map = embedding_host_cache_->host_hash_map_;
  MS_ERROR_IF_NULL(host_hash_map);
  int swap_device_to_host_id = device_to_host_ids[statistics_info_.device_to_host_size_ - 1];
  const auto &hash_id_to_index = host_hash_map->hash_id_to_index();
  const auto &iter = hash_id_to_index.find(swap_device_to_host_id);
  if (iter != hash_id_to_index.end()) {
    auto index = iter->second;
    if (host_hash_map->hash_step(index) != data_step_) {
      host_hash_map->set_hash_step(index, data_step_);
    }
    device_to_host_index[statistics_info_.device_to_host_size_ - 1] = index;
  } else {
    int *host_to_server_index = embedding_host_cache_->host_to_server_index.get();
    int *host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
    while (true) {
      auto index =
        host_hash_map->ParseData(swap_device_to_host_id, host_to_server_index, host_to_server_ids, data_step_,
                                 graph_running_step_, &statistics_info_.host_to_server_size_, &host_need_wait_graph_);
      if (index == INVALID_INDEX_VALUE) {
        RETURN_IF_FALSE(WaitGraphRun());
        continue;
      }
      device_to_host_index[statistics_info_.device_to_host_size_ - 1] = index;
      break;
    }
  }
  return true;
}

void PsCacheManager::LookUpTableTask(size_t indices_lens, size_t outer_dim_size, size_t first_dim_size,
                                     const float *input_addr, const int *indices_addr, float *output_addr) {
  auto type_size = sizeof(float);
  size_t lens = outer_dim_size * type_size;
  for (size_t i = 0; i < indices_lens; ++i) {
    int index = indices_addr[i];
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = index * outer_dim_size;
      auto ret = memcpy_s(output_addr, (indices_lens - i) * lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "LookUpTable task memcpy failed.";
        running_ = false;
        return;
      }
    } else {
      auto ret = memset_s(output_addr, (indices_lens - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(ERROR) << "LookUpTable task memset failed.";
        running_ = false;
        return;
      }
    }
    output_addr += outer_dim_size;
  }
}

bool PsCacheManager::LookUpHostHashTable(size_t embedding_size, size_t indices_lens, const float *hash_table_addr,
                                         const int *indices_addr, float *output_addr) {
  size_t first_dim_size = host_vocab_cache_size_;
  size_t outer_dim_size = embedding_size;

  size_t thread_num = indices_lens / 10000 + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t task_proc_lens = (indices_lens + thread_num - 1) / thread_num;
  size_t i = 0;
  size_t task_offset = 0;
  MS_LOG(DEBUG) << "Indices lens: " << indices_lens << ", one task proc lens:" << task_proc_lens;
  for (; i < thread_num; i++) {
    if (task_offset >= indices_lens) {
      break;
    }
    MS_LOG(DEBUG) << "Task offset: " << task_offset << ", task process lens:" << task_proc_lens;
    threads[i] = std::thread(&PsCacheManager::LookUpTableTask, this, task_proc_lens, outer_dim_size, first_dim_size,
                             hash_table_addr, indices_addr + task_offset, output_addr + task_offset * outer_dim_size);
    task_offset += task_proc_lens;
    if (task_offset + task_proc_lens > indices_lens) {
      task_proc_lens = indices_lens - task_offset;
    }
  }
  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  return running_;
}

bool PsCacheManager::InsertHostHashTable(size_t embedding_size, size_t insert_indices_size, const int *insert_indices,
                                         const float *insert_data, float *hash_table_addr) {
  size_t first_dim_size = host_vocab_cache_size_;
  size_t thread_num = insert_indices_size / 10000 + 1;
  thread_num = thread_num > kMaxThreadNum ? kMaxThreadNum : thread_num;
  std::thread threads[kMaxThreadNum];
  size_t task_proc_lens = (insert_indices_size + thread_num - 1) / thread_num;
  size_t i = 0;
  size_t task_offset = 0;

  auto insert_hash_table_task = [this](size_t insert_indices_size, size_t outer_dim_size, size_t first_dim_size,
                                       const int *insert_indices, const float *insert_data, float *hash_table_addr) {
    auto type_size = sizeof(float);
    size_t copy_len = outer_dim_size * type_size;
    size_t dest_len = copy_len;
    for (size_t i = 0; i < insert_indices_size; ++i) {
      int index = insert_indices[i];
      if (index >= 0 && index < SizeToInt(first_dim_size)) {
        auto ret =
          memcpy_s(hash_table_addr + index * outer_dim_size, dest_len, insert_data + i * outer_dim_size, copy_len);
        if (ret != EOK) {
          MS_LOG(ERROR) << "Insert hash table task memcpy failed.";
          running_ = false;
          return;
        }
      }
    }
  };

  for (; i < thread_num; i++) {
    if (task_offset >= insert_indices_size) {
      break;
    }
    MS_LOG(DEBUG) << "Task offset: " << task_offset << ", task process lens:" << task_proc_lens;
    threads[i] = std::thread(insert_hash_table_task, task_proc_lens, embedding_size, first_dim_size,
                             insert_indices + task_offset, insert_data + task_offset * embedding_size, hash_table_addr);
    task_offset += task_proc_lens;
    if (task_offset + task_proc_lens > insert_indices_size) {
      task_proc_lens = insert_indices_size - task_offset;
    }
  }

  for (size_t j = 0; j < i; j++) {
    threads[j].join();
  }
  return running_;
}

bool PsCacheManager::HashSwapHostToDevice(const HashTableInfo &hash_info) {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_device_cache_->cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);
  auto host_cache_host_to_device_index = embedding_host_cache_->host_to_device_index.get();
  auto device_cache_host_to_device_index = embedding_device_cache_->host_to_device_index.get();
  auto swap_indices_size = statistics_info_.host_to_device_size_;
  if (swap_indices_size == 0) {
    return true;
  }
  auto embedding_size = hash_info.embedding_size;
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);
  RETURN_IF_FALSE(LookUpHostHashTable(embedding_size, swap_indices_size, host_hash_table_addr,
                                      host_cache_host_to_device_index, swap_out_data.get()));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyHostMemToDevice(
    embedding_device_cache_->hash_swap_value_addr_, swap_out_data.get(),
    swap_indices_size * embedding_size * sizeof(float)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyHostMemToDevice(embedding_device_cache_->hash_swap_index_addr_,
                                                                       device_cache_host_to_device_index,
                                                                       swap_indices_size * sizeof(int)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->HashSwapIn(
    hash_table_addr, embedding_device_cache_->hash_swap_value_addr_, embedding_device_cache_->hash_swap_index_addr_,
    cache_vocab_size, embedding_size, swap_indices_size));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->SynchronizeStream());
  return true;
}

bool PsCacheManager::HashSwapDeviceToHost(const HashTableInfo &hash_info) {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_device_cache_->cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_);
  auto swap_indices_size = statistics_info_.device_to_host_size_;
  auto device_cache_device_to_host_index = embedding_device_cache_->device_to_host_index.get();
  auto host_cache_device_to_host_index = embedding_host_cache_->device_to_host_index.get();
  if (swap_indices_size == 0) {
    return true;
  }
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto embedding_size = hash_info.embedding_size;
  auto swap_out_data = std::make_unique<float[]>(swap_indices_size * embedding_size);
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyHostMemToDevice(embedding_device_cache_->hash_swap_index_addr_,
                                                                       device_cache_device_to_host_index,
                                                                       swap_indices_size * sizeof(int)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->HashSwapOut(
    hash_table_addr, embedding_device_cache_->hash_swap_value_addr_, embedding_device_cache_->hash_swap_index_addr_,
    cache_vocab_size, embedding_size, swap_indices_size));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyDeviceMemToHost(
    swap_out_data.get(), embedding_device_cache_->hash_swap_value_addr_,
    swap_indices_size * embedding_size * sizeof(float)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->SynchronizeStream());
  RETURN_IF_FALSE(InsertHostHashTable(embedding_size, IntToSize(swap_indices_size), host_cache_device_to_host_index,
                                      swap_out_data.get(), host_hash_table_addr));
  return true;
}

bool PsCacheManager::HashSwapHostToServer(size_t key, const HashTableInfo &hash_info) {
  MS_ERROR_IF_NULL(embedding_host_cache_);
  auto host_to_server_ids = embedding_host_cache_->host_to_server_ids.get();
  auto host_to_server_index = embedding_host_cache_->host_to_server_index.get();
  auto swap_indices_size = statistics_info_.host_to_server_size_;
  if (swap_indices_size == 0) {
    return true;
  }
  std::vector<int> lookup_ids(swap_indices_size, 0);
  std::vector<float> swap_out_data;
  auto embedding_size = hash_info.embedding_size;
  swap_out_data.resize(swap_indices_size * embedding_size);
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  RETURN_IF_FALSE(LookUpHostHashTable(embedding_size, swap_indices_size, host_hash_table_addr, host_to_server_index,
                                      swap_out_data.data()));

  size_t copy_len = swap_indices_size * sizeof(int);
  size_t dest_len = copy_len;
  auto ret = memcpy_s(lookup_ids.data(), dest_len, host_to_server_ids, copy_len);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Lookup id memcpy failed.";
    return false;
  }
  Worker::GetInstance().UpdateEmbeddingTable({key}, lookup_ids, swap_out_data);
  return true;
}

bool PsCacheManager::HashSwapServerToHost(size_t key, const HashTableInfo &hash_info) {
  MS_ERROR_IF_NULL(embedding_host_cache_);
  auto swap_indices_size = statistics_info_.server_to_host_size_;
  auto server_to_host_ids = embedding_host_cache_->server_to_host_ids.get();
  auto server_to_host_index = embedding_host_cache_->server_to_host_index.get();
  if (swap_indices_size == 0) {
    return true;
  }
  auto host_hash_table_addr = reinterpret_cast<float *>(hash_info.host_address.get());
  auto embedding_size = hash_info.embedding_size;
  std::vector<float> lookup_result(swap_indices_size * embedding_size, 0);
  std::vector<int> lookup_ids(swap_indices_size, 0);
  size_t copy_len = swap_indices_size * sizeof(int);
  size_t dest_len = copy_len;
  auto ret = memcpy_s(lookup_ids.data(), dest_len, server_to_host_ids, copy_len);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Lookup id memcpy failed.";
    return false;
  }
  Worker::GetInstance().DoPSEmbeddingLookup(key, lookup_ids, &lookup_result, mindspore::ps::kEmbeddingLookupCmd);
  RETURN_IF_FALSE(InsertHostHashTable(embedding_size, IntToSize(swap_indices_size), server_to_host_index,
                                      lookup_result.data(), host_hash_table_addr));
  return true;
}

bool PsCacheManager::HashSwapDeviceOut(int *swap_out_index, std::vector<float> *swap_out_data,
                                       const HashTableInfo &hash_info) {
  MS_ERROR_IF_NULL(swap_out_index);
  MS_ERROR_IF_NULL(swap_out_data);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_device_cache_->cache_);
  auto swap_out_index_size = statistics_info_.device_to_host_size_;
  if (swap_out_index_size == 0) {
    return true;
  }
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  auto embedding_size = hash_info.embedding_size;
  swap_out_data->resize(swap_out_index_size * embedding_size);
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyHostMemToDevice(
    embedding_device_cache_->hash_swap_index_addr_, swap_out_index, swap_out_index_size * sizeof(int)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->HashSwapOut(
    hash_table_addr, embedding_device_cache_->hash_swap_value_addr_, embedding_device_cache_->hash_swap_index_addr_,
    cache_vocab_size, embedding_size, swap_out_index_size));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyDeviceMemToHost(
    swap_out_data->data(), embedding_device_cache_->hash_swap_value_addr_,
    swap_out_index_size * embedding_size * sizeof(float)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->RecordEvent());
  return true;
}

bool PsCacheManager::HashSwapDeviceIn(const int *swap_in_ids, const int *swap_in_index, const HashTableInfo &hash_info,
                                      size_t key) {
  MS_ERROR_IF_NULL(swap_in_ids);
  MS_ERROR_IF_NULL(swap_in_index);
  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_device_cache_->cache_);
  auto swap_in_ids_size = statistics_info_.host_to_device_size_;
  if (swap_in_ids_size == 0) {
    return true;
  }
  auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
  auto cache_vocab_size = hash_info.cache_vocab_size;
  auto embedding_size = hash_info.embedding_size;
  // Get id embs by swap_in_ids in host(Pipeline with hash swap-out in device).
  std::vector<float> lookup_result(swap_in_ids_size * embedding_size, 0);
  std::vector<int> lookup_ids(swap_in_ids_size, 0);
  size_t copy_len = swap_in_ids_size * sizeof(int);
  size_t dest_len = copy_len;
  auto ret = memcpy_s(lookup_ids.data(), dest_len, swap_in_ids, copy_len);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Lookup id memcpy failed.";
    return false;
  }
  Worker::GetInstance().DoPSEmbeddingLookup(key, lookup_ids, &lookup_result, mindspore::ps::kEmbeddingLookupCmd);
  // Hash swap-in in device.
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyHostMemToDevice(
    embedding_device_cache_->hash_swap_value_addr_, lookup_result.data(),
    swap_in_ids_size * embedding_size * sizeof(float)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyHostMemToDevice(embedding_device_cache_->hash_swap_index_addr_,
                                                                       swap_in_index, swap_in_ids_size * sizeof(int)));
  RETURN_IF_FALSE(embedding_device_cache_->cache_->HashSwapIn(
    hash_table_addr, embedding_device_cache_->hash_swap_value_addr_, embedding_device_cache_->hash_swap_index_addr_,
    cache_vocab_size, embedding_size, swap_in_ids_size));
  return true;
}

bool PsCacheManager::UpdataEmbeddingTable(const std::vector<float> &swap_out_data, int *const swap_out_ids,
                                          size_t key) {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  MS_ERROR_IF_NULL(embedding_device_cache_->cache_);
  MS_ERROR_IF_NULL(swap_out_ids);
  auto swap_out_ids_size = statistics_info_.device_to_host_size_;
  if (swap_out_ids_size == 0) {
    return true;
  }
  std::vector<int> lookup_ids(swap_out_ids_size, 0);
  size_t copy_len = swap_out_ids_size * sizeof(int);
  size_t dest_len = copy_len;
  auto ret = memcpy_s(lookup_ids.data(), dest_len, swap_out_ids, copy_len);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Lookup id memcpy failed.";
    return false;
  }
  // Need synchronize event to ensure that the swap-out in device is completed.
  RETURN_IF_FALSE(embedding_device_cache_->cache_->SynchronizeEvent());
  Worker::GetInstance().UpdateEmbeddingTable({key}, lookup_ids, swap_out_data);
  return true;
}

void PsCacheManager::SyncEmbeddingTable() {
  if (finish_embedding_table_sync_) {
    return;
  }
  if (!initialized_ps_cache_) {
    return;
  }
  if (!SyncHostEmbeddingTable()) {
    MS_LOG(ERROR) << "SyncHostEmbeddingTable failed.";
  }
  if (!SyncDeviceEmbeddingTable()) {
    MS_LOG(ERROR) << "SyncDeviceEmbeddingTable failed.";
  }
  finish_embedding_table_sync_ = true;
}

bool PsCacheManager::SyncHostEmbeddingTable() {
  MS_ERROR_IF_NULL(embedding_host_cache_);
  MS_ERROR_IF_NULL(embedding_host_cache_->host_hash_map_);
  const auto &hash_id_to_index = embedding_host_cache_->host_hash_map_->hash_id_to_index();
  size_t swap_indices_lens = hash_id_to_index.size();
  if (swap_indices_lens == 0) {
    return true;
  }
  std::unique_ptr<int[]> host_to_server_ids_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(host_to_server_ids_ptr);
  std::unique_ptr<int[]> host_to_server_indices_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(host_to_server_indices_ptr);
  size_t idx = 0;
  for (const auto &item : hash_id_to_index) {
    host_to_server_ids_ptr[idx] = item.first;
    host_to_server_indices_ptr[idx++] = item.second;
  }
  for (const auto &item : hash_tables_) {
    const auto &hash_info = item.second;
    if (hash_info.param_init_info_.param_type_ != kWeight) {
      continue;
    }
    auto key = Worker::GetInstance().GetParamKey(item.first);
    std::vector<int> lookup_ids(swap_indices_lens, 0);
    std::vector<float> swap_out_data;
    auto embedding_size = hash_info.embedding_size;
    swap_out_data.resize(swap_indices_lens * embedding_size);
    auto host_hash_table_addr = hash_info.host_address.get();
    MS_ERROR_IF_NULL(host_hash_table_addr);
    RETURN_IF_FALSE(LookUpHostHashTable(embedding_size, swap_indices_lens, host_hash_table_addr,
                                        host_to_server_indices_ptr.get(), swap_out_data.data()));

    size_t copy_len = swap_indices_lens * sizeof(int);
    size_t dest_len = copy_len;
    auto ret = memcpy_s(lookup_ids.data(), dest_len, host_to_server_ids_ptr.get(), copy_len);
    if (ret != EOK) {
      MS_LOG(ERROR) << "Lookup id memcpy failed.";
      return false;
    }
    Worker::GetInstance().UpdateEmbeddingTable({key}, lookup_ids, swap_out_data);
  }
  return true;
}

bool PsCacheManager::SyncDeviceEmbeddingTable() {
  MS_ERROR_IF_NULL(embedding_device_cache_);
  const auto &device_hash_map = embedding_device_cache_->device_hash_map_;
  MS_ERROR_IF_NULL(device_hash_map);
  const auto &hash_id_to_index = device_hash_map->hash_id_to_index();
  size_t swap_indices_lens = hash_id_to_index.size();
  if (swap_indices_lens == 0) {
    return true;
  }
  std::unique_ptr<int[]> device_to_server_ids_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(device_to_server_ids_ptr);
  std::unique_ptr<int[]> device_to_server_indices_ptr = std::make_unique<int[]>(swap_indices_lens);
  MS_ERROR_IF_NULL(device_to_server_indices_ptr);
  size_t idx = 0;
  for (const auto &item : hash_id_to_index) {
    device_to_server_ids_ptr[idx] = item.first;
    device_to_server_indices_ptr[idx++] = item.second;
  }
  for (const auto &item : hash_tables_) {
    const auto &hash_info = item.second;
    if (hash_info.param_init_info_.param_type_ != kWeight) {
      continue;
    }
    auto key = Worker::GetInstance().GetParamKey(item.first);
    std::vector<int> lookup_ids(swap_indices_lens, 0);
    std::vector<float> swap_out_data;
    auto embedding_size = hash_info.embedding_size;
    swap_out_data.resize(swap_indices_lens * embedding_size);
    std::unique_ptr<float[]> device_hash_table_addr_tmp =
      std::make_unique<float[]>(device_hash_map->hash_capacity() * embedding_size);
    MS_ERROR_IF_NULL(device_hash_table_addr_tmp);

    auto hash_table_addr = reinterpret_cast<float *>(hash_info.device_address.addr);
    MS_ERROR_IF_NULL(hash_table_addr);
    auto hash_table_size = hash_info.device_address.size;
    RETURN_IF_FALSE(embedding_device_cache_->cache_->CopyDeviceMemToHost(device_hash_table_addr_tmp.get(),
                                                                         hash_table_addr, hash_table_size));
    RETURN_IF_FALSE(embedding_device_cache_->cache_->SynchronizeStream());
    RETURN_IF_FALSE(LookUpHostHashTable(embedding_size, swap_indices_lens, device_hash_table_addr_tmp.get(),
                                        device_to_server_indices_ptr.get(), swap_out_data.data()));

    size_t copy_len = swap_indices_lens * sizeof(int);
    size_t dest_len = copy_len;
    auto ret = memcpy_s(lookup_ids.data(), dest_len, device_to_server_ids_ptr.get(), copy_len);
    if (ret != EOK) {
      MS_LOG(ERROR) << "Lookup id memcpy failed.";
      return false;
    }
    Worker::GetInstance().UpdateEmbeddingTable({key}, lookup_ids, swap_out_data);
  }
  return true;
}

void PsCacheManager::DumpHashTables(bool dump_device_tables) const {
  for (const auto &item : hash_tables_) {
    const auto &param_name = item.first;
    size_t cache_vocab_size = item.second.cache_vocab_size;
    size_t host_cache_vocab_size = item.second.host_cache_vocab_size;
    size_t embedding_size = item.second.embedding_size;
    size_t vocab_size = item.second.vocab_size;
    MS_LOG(INFO) << "Hash table info:"
                 << " embedding table name:" << param_name << ", vocab size:" << vocab_size
                 << ", embedding size:" << embedding_size << ", device cache size:" << cache_vocab_size
                 << ", host cache size:" << host_cache_vocab_size
                 << ", device cache address:" << reinterpret_cast<void *>(item.second.device_address.addr)
                 << ", host cache address:" << reinterpret_cast<void *>(item.second.host_address.get());
    if (dump_device_tables) {
      std::unique_ptr<float[]> output = std::make_unique<float[]>(item.second.device_address.size / 4);
      embedding_device_cache_->cache_->CopyDeviceMemToHost(output.get(), item.second.device_address.addr,
                                                           item.second.device_address.size);
      embedding_device_cache_->cache_->SynchronizeStream();
      for (size_t i = 0; i < cache_vocab_size; i++) {
        for (size_t j = 0; j < embedding_size; j++) {
          std::cout << output[i * embedding_size + j] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
}

void PsCacheManager::DumpStatisticsInfo(size_t each_print_step) {
  // Default each 1000 step prints ps cache hit rate.
  if (data_step_ % each_print_step == 0) {
    statistics_info_.batch_id_unique_count_ = statistics_info_.hash_hit_count_ + statistics_info_.host_to_device_size_;
    auto repeat_rate = SizeToFloat(statistics_info_.batch_id_count_ - statistics_info_.batch_id_unique_count_) /
                       statistics_info_.batch_id_count_;
    auto device_hit_rate = SizeToFloat(statistics_info_.hash_hit_count_) / statistics_info_.batch_id_unique_count_;
    auto host_hit_rate = SizeToFloat(statistics_info_.batch_id_unique_count_ - statistics_info_.server_to_host_size_) /
                         statistics_info_.batch_id_unique_count_;
    MS_LOG(INFO) << "PS embedding cache data statistics info(total id num:" << statistics_info_.batch_id_count_
                 << ", unique id num:" << statistics_info_.batch_id_unique_count_
                 << ", host swap to device num:" << statistics_info_.host_to_device_size_
                 << ", device swap to host num:" << statistics_info_.device_to_host_size_
                 << ", host swap to server num:" << statistics_info_.host_to_server_size_
                 << ", server swap to host num:" << statistics_info_.server_to_host_size_
                 << ", data repeat rate:" << repeat_rate * 100 << "%, device cache hit rate:" << device_hit_rate * 100
                 << "%, host cache hit rate:" << host_hit_rate * 100 << ").";
  }
}
}  // namespace ps
}  // namespace mindspore
