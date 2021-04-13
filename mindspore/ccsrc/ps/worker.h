/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_WORKER_H_
#define MINDSPORE_CCSRC_PS_WORKER_H_

#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <functional>
#include <algorithm>
#include <map>
#include <mutex>
#include <unordered_set>
#include <unordered_map>

#include "utils/log_adapter.h"
#include "ir/tensor.h"
#include "ps/util.h"
#include "ps/constants.h"
#include "utils/shape_utils.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/core/worker_node.h"
#include "ps/embedding_table_shard_metadata.h"
#include "proto/comm.pb.h"
#include "proto/ps.pb.h"
#include "ps/ps_context.h"

namespace mindspore {
namespace ps {
class Worker {
 public:
  static Worker &GetInstance() {
    static Worker instance;
    return instance;
  }
  using Callback = std::function<void()>;
  using PartitionEmbeddingMessages = std::vector<std::pair<bool, EmbeddingTableLookup>>;
  using PartitionKVMessages = std::vector<std::pair<bool, KVMessage>>;

  using EmbeddingPartitioner = std::function<void(
    const EmbeddingTableLookup &send, PartitionEmbeddingMessages *partition, const std::map<int64_t, int64_t> &attrs)>;
  using KVPartitioner =
    std::function<void(const KVMessage &send, PartitionKVMessages *partition, const std::map<int64_t, int64_t> &attrs)>;

  void Run();
  void Push(const std::vector<size_t> &keys, std::vector<uintptr_t> addrs, const ShapeVector &sizes);
  void Pull(const size_t key, void *dev_addr, const size_t size);
  size_t SetParamKey(const std::string &param_name);
  size_t GetParamKey(const std::string &param_name);
  void SetParamInitInServer(const std::string &param_name, bool init_in_server);
  bool GetParamInitInServer(const std::string &param_name);
  void SetKeyOptimId(size_t key, const std::string &optimizer_name);
  void SetOptimInputShapes(size_t key, const ShapeVector &shape);
  void AddEmbeddingTable(const Key &key, const size_t &row_count);
  void InitPSEmbeddingTable(const size_t &key, const std::vector<size_t> &input_shape,
                            const std::vector<size_t> &indices_shape, const std::vector<size_t> &output_shape,
                            const ParamInitInfoMessage &info);
  void InitPSParamAndOptim(const AnfNodePtr &input_node, const tensor::TensorPtr &tensor);
  void DoPSEmbeddingLookup(const Key &key, const std::vector<int> &lookup_ids, std::vector<float> *lookup_result,
                           int64_t cmd);
  void UpdateEmbeddingTable(const std::vector<Key> &keys, const std::vector<int> &lookup_ids,
                            const std::vector<float> &vals);

  bool running() { return running_; }
  void Finalize();

 private:
  Worker() : server_num_(-1), running_(false), key_cnt_(0) {}
  ~Worker() = default;
  Worker(const Worker &) = delete;
  Worker &operator=(const Worker &) = delete;

  void Initialize();
  bool IsKeyInit(const size_t key);
  void AddKeyToServerId(const Key &key);
  void AddKeyByHashMod(const Key &key);
  void InitPSOptimId(const size_t param_key);
  void InitPSOptimInputShapes(const size_t key);
  void InitPSParamData(const std::vector<size_t> &keys, void *const origin_addr, size_t size);
  bool IsReadyForPush(const Key &key);
  bool IsReadyForPull(const Key &key);
  void PrepareSparseGradient(const size_t begin, const size_t end, const std::unordered_set<int> &distinct_ids,
                             const std::vector<std::pair<int, float *>> &indice_to_grads, const int *all_indice,
                             const size_t segment_size, float *gradient, int *indices);
  void BuildSparseValue(const std::vector<int> &lengths, const size_t grad_index, const size_t indice_index,
                        const float *original_data, const float *grads, int *indices, std::vector<float> *reduced_data);

  void PushData(const std::vector<Key> &keys, const std::vector<float> &vals, const std::vector<int> &lens = {},
                int command = 0, int64_t priority = 0);
  void PushSparseData(const std::vector<Key> &keys, const std::vector<float> &vals, const std::vector<int> &lens,
                      size_t grad_index, size_t indice_index, size_t first_dim_size, size_t outer_dim_size);
  void PullData(const std::vector<Key> &keys, std::vector<float> *const vals, std::vector<int> *lens = nullptr,
                int cmd = 0, int64_t priority = 0);

  void LookupIdPartitioner(const EmbeddingTableLookup &send, PartitionEmbeddingMessages *partition,
                           const std::map<int64_t, int64_t> &attrs);

  void SparsePartitioner(const KVMessage &send, PartitionKVMessages *partition,
                         const std::map<int64_t, int64_t> &attrs);
  void RoundRobinPartitioner(const KVMessage &send, PartitionKVMessages *partition,
                             const std::map<int64_t, int64_t> &attrs);
  void WorkerInitEmbeddingPartitioner(const KVMessage &send, std::vector<std::pair<bool, KVMessage>> *partition,
                                      const std::map<int64_t, int64_t> &attrs);
  void UpdateEmbeddingPartitioner(const KVMessage &send, PartitionKVMessages *partition,
                                  const std::map<int64_t, int64_t> &attrs);
  void BroadcastPartitioner(const KVMessage &send, PartitionKVMessages *partition,
                            const std::map<int64_t, int64_t> &attrs);
  void SendForPush(int cmd, const KVMessage &send, const KVPartitioner &partitioner,
                   const std::map<int64_t, int64_t> &attrs);
  void SendForPull(int cmd, const KVMessage &send, const KVPartitioner &partitioner,
                   const std::map<int64_t, int64_t> &attrs, std::vector<float> *vals, std::vector<int> *lens);

  int64_t server_num_;
  bool running_;
  std::mutex running_mutex_;
  size_t key_cnt_;
  std::map<std::string, size_t> param_to_key_;
  std::map<size_t, bool> init_keys_;
  std::map<size_t, int64_t> key_to_optimId_;
  std::map<size_t, std::vector<ShapeVector>> key_to_optim_shapes_;
  std::map<std::string, bool> param_to_init_in_server_;
  core::WorkerNode worker_node_;

  EmbeddingPartitioner lookup_partitioner_;
  KVPartitioner sparse_partitioner_;
  KVPartitioner round_robin_partitioner_;
  KVPartitioner worker_init_embedding_partitioner_;
  KVPartitioner update_embedding_partitioner_;
  KVPartitioner broadcast_partitioner_;
  std::unordered_map<Key, int64_t> key_to_server_id_;
  std::unordered_map<Key, size_t> embedding_row_cnt_;

  std::unordered_map<Key, std::shared_ptr<std::vector<EmbeddingTableShardMetadata>>> embedding_table_ranges_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_WORKER_H_
