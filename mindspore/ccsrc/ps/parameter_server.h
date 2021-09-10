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

#ifndef MINDSPORE_CCSRC_PS_PARAMETER_SERVER_H_
#define MINDSPORE_CCSRC_PS_PARAMETER_SERVER_H_

#include <unistd.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cmath>
#include <random>
#include <utility>
#include <list>
#include <map>
#include <functional>
#include <algorithm>

#include "ir/func_graph.h"
#include "backend/session/session_basic.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/session_factory.h"
#include "ps/optimizer_info.h"
#include "ps/optimizer_info_builder.h"
#include "ps/ps_context.h"
#include "runtime/device/cpu/kernel_select_cpu.h"
#include "utils/ms_context.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/ps/pserver_kernel.h"
#include "backend/kernel_compiler/cpu/ps/sparse_apply_adam_ps_kernel.h"
#include "backend/kernel_compiler/cpu/ps/sparse_apply_lazy_adam_ps_kernel.h"
#include "backend/kernel_compiler/cpu/ps/sparse_apply_ftrl_ps_kernel.h"
#include "backend/kernel_compiler/cpu/ps/apply_momentum_ps_kernel.h"
#include "backend/kernel_compiler/cpu/ps/embedding_look_up_ps_kernel.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "ps/random_normal/random_normal.h"

#include "ps/constants.h"
#include "ps/util.h"
#include "ps/embedding_table_shard_metadata.h"
#include "utils/log_adapter.h"
#include "proto/comm.pb.h"
#include "proto/ps.pb.h"
#include "ps/core/server_node.h"
#include "ps/core/node.h"

namespace mindspore {
namespace ps {
class ParameterServer {
 public:
  static ParameterServer &GetInstance() {
    static ParameterServer instance;
    return instance;
  }

  void Run(const FuncGraphPtr &func_graph);

 private:
  ParameterServer()
      : pserver_num_(0),
        worker_num_(0),
        grad_accum_count_(0),
        handler_(nullptr),
        func_graph_(nullptr),
        sess_(nullptr),
        running_(true),
        thread_(nullptr),
        server_node_(nullptr) {}
  ~ParameterServer() = default;
  ParameterServer(const ParameterServer &) = delete;
  ParameterServer &operator=(const ParameterServer &) = delete;

  class ServerHandler {
   public:
    explicit ServerHandler(ParameterServer *ps) : ps_(ps) {}
    ~ServerHandler() = default;
    void Init();
    void operator()(const std::shared_ptr<core::TcpConnection> &conn, const std::shared_ptr<core::MessageMeta> &meta,
                    const DataPtr &data, size_t size);
    void HandlePushReq(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandlePullReq(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleInitWeights(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleInitWeightToOptimId(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleInitInputsShape(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleInitEmbeddings(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleCheckReadyForPush(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleCheckReadyForPull(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleEmbeddingLookup(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleUpdateEmbeddings(const DataPtr &data, size_t size, const VectorPtr &res);
    void HandleFinalize(const DataPtr &data, size_t size, const VectorPtr &res);

   private:
    ParameterServer *ps_;
    typedef void (ServerHandler::*RequestHandler)(const DataPtr &data, size_t size, const VectorPtr &res);
    std::unordered_map<int, RequestHandler> handlers_;
    std::unordered_map<int, std::string> commands_;
    std::unordered_map<Key, bool> init_weights_;
    std::unordered_map<Key, bool> init_weight_to_optim_;
    std::unordered_map<Key, bool> init_optim_info_;
  };

  bool Init(const FuncGraphPtr &func_graph);
  void InitOptimInfoBuilders();
  void InitWeightKeyToOptims(const Key &key, const int64_t &optim_id);
  void InitOptimInputsShape(const Keys &keys, const Values &values, const Lengths &lengths);
  void InitWeight(const Key &key, const WeightPtr &weight);
  void InitGrad(const Key &key, const GradPtr &grad);
  void InitEmbeddingTable(const Key &key,
                          const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes,
                          const ParamInitInfo &param_init_info);
  bool HasWeight(const Key &key);
  void Finalize();
  void UpdateWeights();
  void AccumGrad(const Keys &key, const Values &values, const Lengths &lengths);
  WeightPtr weight(const Key &key);
  void DoEmbeddingLookup(Key key, const LookupIds &lookup_ids, KVMessage *res);
  void UpdateEmbeddings(const Key &key, const LookupIds &lookup_ids, const Values &vals);
  inline bool ReadyForUpdateWeights() const;
  inline bool ReadyForPush(const Key &key);
  inline bool ReadyForPull(const Key &key);
  inline void ResetGradAccumCount();
  const CNodePtr GetCNode(const std::string &name) const;
  inline std::mutex &mutex();
  void GetEmbeddingTableParamPtr();
  void SyncEmbeddingTables();
  // Cache embedding table parameter by map, key: parameter name, value: parameter node pointer
  void CacheEmbeddingTableParamPtr();

  size_t pserver_num_;
  size_t worker_num_;
  size_t grad_accum_count_;
  std::unique_ptr<ServerHandler> handler_;
  FuncGraphPtr func_graph_;
  std::shared_ptr<session::SessionBasic> sess_;
  bool running_;
  bool embedding_param_ptr_cached_{false};
  // Used to cache embedding table parameter, key: parameter name, value: parameter node pointer
  std::map<std::string, ParameterPtr> embedding_parameter_tables_;

  std::unordered_map<Key, std::shared_ptr<PServerKernel>> optimizers_;
  std::unordered_map<Key, InputsShapePtr> optim_inputs_shape_;
  std::unordered_map<Key, InputsShapePtr> original_optim_inputs_shape_;
  std::unordered_map<Key, std::shared_ptr<OptimizerInfo>> optim_infos_;
  std::unordered_map<std::string, std::shared_ptr<OptimizerInfoBuilder>> optim_info_builders_;
  std::unordered_map<Key, std::string> weight_key_to_optims_;
  std::unordered_map<Key, std::string> weight_key_to_optim_op_;
  std::unordered_map<Key, WeightPtr> weights_;
  std::unordered_map<Key, bool> is_embedding_;
  std::unordered_map<Key, WeightPtr> grads_;
  std::unordered_map<Key, size_t> grads_accum_counter_;
  std::unordered_map<Key, std::shared_ptr<PServerKernel>> embedding_lookup_ops_;
  std::unordered_map<Key, uint64_t> tokens_;

  std::mutex mutex_;
  std::condition_variable apply_grads_cv_;

  std::unique_ptr<std::thread> thread_;
  std::shared_ptr<core::ServerNode> server_node_;
  std::map<Key, ParameterPtr> embedding_tables_;

  friend class ServerHandler;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PARAMETER_SERVER_H_
