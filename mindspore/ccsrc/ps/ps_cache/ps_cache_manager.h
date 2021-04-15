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

#ifndef MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_MANAGER_H_
#define MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_MANAGER_H_

#include <map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <utility>
#include <memory>
#include <condition_variable>
#include "utils/ms_context.h"
#include "backend/kernel_compiler/kernel.h"
#include "utils/shape_utils.h"
#include "ir/tensor.h"
#include "ps/constants.h"
#include "ps/worker.h"
#include "ps/ps_context.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/ps_cache/embedding_hash_map.h"
#include "ps/ps_cache/ps_cache_factory.h"

namespace mindspore {
namespace ps {
constexpr size_t kHostCacheScaleFactor = 10;
constexpr size_t kMaxThreadNum = 16;
constexpr size_t kMinIdsPerThread = 10000;
using mindspore::kernel::Address;

struct HashTableInfo {
  size_t cache_vocab_size{0};
  size_t host_cache_vocab_size{0};
  size_t embedding_size{0};
  size_t vocab_size{0};
  Address device_address{nullptr, 0};
  std::shared_ptr<float[]> host_address{nullptr};
  ParamInitInfo param_init_info_;
};

struct EmbeddingDeviceCache {
  EmbeddingDeviceCache(size_t batch_elements, size_t cache_vocab_size)
      : hash_swap_index_addr_(nullptr), hash_swap_value_addr_(nullptr) {
    device_to_host_index = std::make_unique<int[]>(batch_elements);
    device_to_host_ids = std::make_unique<int[]>(batch_elements);
    host_to_device_index = std::make_unique<int[]>(batch_elements);
    host_to_device_ids = std::make_unique<int[]>(batch_elements);
    device_hash_map_ = std::make_shared<EmbeddingHashMap>(0, cache_vocab_size);
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    auto devcie_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    cache_ = PsCacheFactory::Get().ps_cache(devcie_target);
  }
  std::unique_ptr<int[]> device_to_host_index;
  std::unique_ptr<int[]> device_to_host_ids;
  std::unique_ptr<int[]> host_to_device_index;
  std::unique_ptr<int[]> host_to_device_ids;
  int *hash_swap_index_addr_;
  float *hash_swap_value_addr_;
  std::shared_ptr<EmbeddingHashMap> device_hash_map_;
  std::shared_ptr<PsCacheBasic> cache_;
};

struct EmbeddingHostCache {
  EmbeddingHostCache(size_t batch_elements, size_t host_cache_vocab_size) {
    host_to_server_index = std::make_unique<int[]>(batch_elements);
    host_to_server_ids = std::make_unique<int[]>(batch_elements);
    server_to_host_index = std::make_unique<int[]>(batch_elements);
    server_to_host_ids = std::make_unique<int[]>(batch_elements);
    host_to_device_index = std::make_unique<int[]>(batch_elements);
    device_to_host_index = std::make_unique<int[]>(batch_elements);
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

struct PsCacheStatisticsInfo {
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

class PsCacheManager {
 public:
  static PsCacheManager &GetInstance() {
    static PsCacheManager instance;
    return instance;
  }
  void Initialize();
  void InsertHashTableSize(const std::string &param_name, size_t cache_vocab_size, size_t embedding_size,
                           size_t vocab_size);
  void InsertWeightInitInfo(const std::string &param_name, size_t global_seed, size_t op_seed);
  void InsertAccumuInitInfo(const std::string &param_name, float init_val);
  void ReInsertHashTableSize(const std::string &new_param_name, const std::string &cur_param_name,
                             size_t cache_vocab_size, size_t embedding_size);
  void CloneHashTable(const std::string &dest_param_name, const std::string &src_param_name);
  const Address &QueryHashTableAddr(const std::string &param_name) const;
  const size_t &QueryHashTableSize(const std::string &param_name) const;
  bool IsHashTable(const std::string &param_name) { return hash_tables_.count(param_name) != 0; }
  void set_batch_elements(size_t batch_elements) { batch_elements_ = batch_elements; }
  void set_rank_id(int rank_id) { rank_id_ = rank_id; }
  bool initialized_ps_cache() const { return initialized_ps_cache_; }
  size_t vocab_cache_size() const { return vocab_cache_size_; }
  int cache_indices_lower_bound() const;
  void DoProcessData(uint32_t device_id, const void *context);
  void IncreaseGraphStep(const std::string &channel_name);
  void SyncEmbeddingTable();
  void Finalize();
  void DumpHashTables(bool dump_device_tables = false) const;

 private:
  PsCacheManager() = default;
  ~PsCacheManager() = default;
  PsCacheManager(const PsCacheManager &) = delete;
  PsCacheManager &operator=(const PsCacheManager &) = delete;
  bool IncreaseStep();
  void set_current_graph_step() { graph_running_step_ = graph_step_; }
  std::string channel_name();
  void set_channel_name(const std::string channel_name);
  void InitParameterServer();
  void InitDataChannel();
  void AllocMemForHashTable();
  void SetLocalIdRank();
  void ProcessDataTask(uint32_t device_id, const void *context);
  bool ProcessData();
  bool ParseData(const int *batch_ids, const size_t batch_ids_len, int *hash_index);
  bool WaitGraphRun();
  bool ParseDeviceData(size_t id, bool *need_swap_device_to_host, bool *need_swap_host_to_device, int *hash_index);
  bool ParseHostDataHostToDevice(size_t id);
  bool ParseHostDataDeviceToHost();
  bool HashSwapDeviceOut(int *swap_out_index, std::vector<float> *swap_out_data, const HashTableInfo &hash_info);
  bool HashSwapDeviceIn(const int *swap_in_ids, const int *swap_in_index, const HashTableInfo &hash_info, size_t key);
  bool HashSwapHostToDevice(const HashTableInfo &hash_info);
  bool HashSwapDeviceToHost(const HashTableInfo &hash_info);
  bool HashSwapHostToServer(size_t key, const HashTableInfo &hash_info);
  bool HashSwapServerToHost(size_t key, const HashTableInfo &hash_info);
  bool InsertHostHashTable(size_t embedding_size, size_t insert_indices_size, const int *insert_indices,
                           const float *insert_data, float *hash_table_addr);
  bool LookUpHostHashTable(size_t embedding_size, size_t indices_lens, const float *hash_table_addr,
                           const int *indices_addr, float *output_addr);
  bool UpdataEmbeddingTable(const std::vector<float> &swap_out_data, int *const swap_out_ids, size_t key);
  void LookUpTableTask(size_t indices_lens, size_t outer_dim_size, size_t first_dim_size, const float *input_addr,
                       const int *indices_addr, float *output_addr);
  bool CheckFinishInsertInitInfo() const;
  void AddEmbeddingTable() const;
  void DumpStatisticsInfo(size_t each_print_step = 1000);
  bool SyncHostEmbeddingTable();
  bool SyncDeviceEmbeddingTable();
  bool CheckCacheHitOrOutRangeTask(const int *batch_ids, const size_t batch_ids_len, int *hash_index, bool *in_device,
                                   bool *out_range, size_t *hash_hit_count);
  bool CheckCacheHitOrOutRange(const int *batch_ids, const size_t batch_ids_len, int *hash_index, bool *in_device,
                               bool *out_range);
  bool ResetEmbeddingHashMap();

  bool initialized_ps_cache_{false};
  std::string channel_name_;
  std::mutex channel_mutex_;
  std::atomic_ulong graph_step_{0};
  size_t graph_running_step_{0};
  size_t data_step_{0};
  std::mutex data_mutex_;
  std::condition_variable data_prase_;
  std::condition_variable insert_init_info_;
  std::thread process_data_thread_;

  std::map<std::string, HashTableInfo> hash_tables_;
  std::shared_ptr<EmbeddingDeviceCache> embedding_device_cache_;
  std::shared_ptr<EmbeddingHostCache> embedding_host_cache_;

  size_t vocab_size_{0};
  size_t vocab_cache_size_{0};
  size_t host_vocab_cache_size_{0};
  size_t batch_elements_{0};
  PsCacheStatisticsInfo statistics_info_;
  std::pair<int, int> emb_table_slice_bounds_;
  std::pair<int, int> cache_indices_bounds_;
  int vocab_cache_size_diff_{0};
  int rank_id_{0};
  std::atomic_bool finish_insert_init_info_{false};
  std::atomic_bool finish_init_parameter_server_{false};
  std::atomic_bool running_{false};
  bool finish_embedding_table_sync_{false};
  bool device_need_wait_graph_{false};
  bool host_need_wait_graph_{false};
};

static PsCacheManager &ps_cache_instance = PsCacheManager::GetInstance();
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_MANAGER_H_
