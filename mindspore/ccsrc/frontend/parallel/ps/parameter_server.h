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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_PARAMETER_SERVER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_PARAMETER_SERVER_H_

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
#include <list>
#include "ir/func_graph.h"
#include "backend/session/session_basic.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/session_factory.h"
#include "frontend/parallel/ps/common.h"
#include "frontend/parallel/ps/optimizer_info.h"
#include "frontend/parallel/ps/optimizer_info_builder.h"
#include "frontend/parallel/ps/util.h"
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

namespace mindspore {
namespace parallel {
namespace ps {
using mindspore::kernel::ps::PServerKernel;
template <typename T>
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
        rank_id_(0),
        grad_accum_count_(0),
        ps_(new ::ps::KVServer<T>(0)),
        handler_(nullptr),
        func_graph_(nullptr),
        sess_(nullptr),
        running_(true),
        thread_(nullptr) {}
  ~ParameterServer() = default;
  ParameterServer(const ParameterServer &) = delete;
  ParameterServer &operator=(const ParameterServer &) = delete;

  class ServerHandler {
   public:
    explicit ServerHandler(ParameterServer *ps) : ps_(ps) {}
    void Init();
    void operator()(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVServer<T> *server);

   private:
    void HandlePushReq(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandlePullReq(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleInitWeights(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleInitWeightToOptimId(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                   ::ps::KVPairs<T> *res);
    void HandleInitInputsShape(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleInitEmbeddings(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleCheckReadyForPush(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleCheckReadyForPull(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleEmbeddingLookup(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleFinalize(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);

    ParameterServer *ps_;
    typedef void (ServerHandler::*RequestHandler)(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                  ::ps::KVPairs<T> *res);
    std::unordered_map<int, RequestHandler> handlers_;
    std::unordered_map<Key, bool> init_weights_;
    std::unordered_map<Key, bool> init_weight_to_optim_;
    std::unordered_map<Key, bool> init_optim_info_;
  };

  bool Init(const FuncGraphPtr &func_graph);
  void InitOptimInfoBuilders();
  void InitWeightKeyToOptims(const Key &key, const int &optim_id);
  void InitOptimInputsShape(const Keys &keys, const Values &values, const Lengths &lengths);
  void InitWeight(const Key &key, const WeightPtr &weight);
  void InitGrad(const Key &key, const GradPtr &grad);
  void InitEmbeddingTable(const Key &key,
                          const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes);
  void Finalize();
  void UpdateWeights();
  void AccumGrad(const Keys &key, const Values &values, const Lengths &lengths);
  WeightPtr weight(const Key &key);
  void DoEmbeddingLookup(Key key, const LookupIds &lookup_ids, ::ps::KVPairs<T> *res);
  int SumOfShapes(const std::vector<int> &shapes) const;
  bool ReadyForUpdateWeights();
  bool ReadyForPush(const Key &key);
  bool ReadyForPull(const Key &key);
  void ResetGradAccumCount();
  const CNodePtr GetCNode(const std::string &name) const;
  std::mutex &mutex();

  size_t pserver_num_;
  size_t worker_num_;
  size_t rank_id_;
  size_t grad_accum_count_;
  std::unique_ptr<::ps::KVServer<T>> ps_;
  std::unique_ptr<ServerHandler> handler_;
  FuncGraphPtr func_graph_;
  std::shared_ptr<session::SessionBasic> sess_;
  bool running_;

  std::unordered_map<Key, std::shared_ptr<PServerKernel>> optimizers_;
  std::unordered_map<Key, InputsShapePtr> optim_inputs_shape_;
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

  friend class ServerHandler;
};

class FuncGraph;
template <typename T>
void ParameterServer<T>::ServerHandler::operator()(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                   ::ps::KVServer<T> *server) {
  ::ps::KVPairs<T> res;
  if (handlers_.count(req_meta.cmd) > 0) {
    auto &handler_ptr = handlers_[req_meta.cmd];
    (this->*handler_ptr)(req_meta, req_data, &res);
  } else if (req_meta.push) {
    HandlePushReq(req_meta, req_data, &res);
  } else {
    HandlePullReq(req_meta, req_data, &res);
  }
  server->Response(req_meta, res);
}

template <typename T>
void ParameterServer<T>::ServerHandler::Init() {
  handlers_[kInitWeightsCmd] = &ServerHandler::HandleInitWeights;
  handlers_[kInitWeightToOptimIdCmd] = &ServerHandler::HandleInitWeightToOptimId;
  handlers_[kInitOptimInputsShapeCmd] = &ServerHandler::HandleInitInputsShape;
  handlers_[kInitEmbeddingsCmd] = &ServerHandler::HandleInitEmbeddings;
  handlers_[kCheckReadyForPushCmd] = &ServerHandler::HandleCheckReadyForPush;
  handlers_[kCheckReadyForPullCmd] = &ServerHandler::HandleCheckReadyForPull;
  handlers_[kEmbeddingLookupCmd] = &ServerHandler::HandleEmbeddingLookup;
  handlers_[kFinalizeCmd] = &ServerHandler::HandleFinalize;
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandlePushReq(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                      ::ps::KVPairs<T> *res) {
  ps_->AccumGrad(req_data.keys, req_data.vals, req_data.lens);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandlePullReq(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                      ::ps::KVPairs<T> *res) {
  res->keys = req_data.keys;
  ::ps::Key key = req_data.keys[0];
  res->vals = *(ps_->weight(key));
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleInitWeights(const ::ps::KVMeta &req_meta,
                                                          const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  size_t key_num = req_data.keys.size();
  T *data_ptr = req_data.vals.data();
  size_t pos = 0;
  for (size_t i = 0; i < key_num; i++) {
    Key key = req_data.keys[i];
    size_t data_len = req_data.lens.size() != key_num ? req_data.vals.size() / key_num : req_data.lens[i];

    WeightPtr weight_ptr = std::make_shared<::ps::SArray<T>>();
    weight_ptr->CopyFrom(data_ptr + pos, data_len);
    ps_->InitWeight(key, weight_ptr);

    GradPtr grad_ptr = std::make_shared<::ps::SArray<T>>(data_len, 0);
    ps_->InitGrad(key, grad_ptr);
    pos += data_len;
  }
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleInitWeightToOptimId(const ::ps::KVMeta &req_meta,
                                                                  const ::ps::KVPairs<T> &req_data,
                                                                  ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  size_t key_num = req_data.keys.size();
  for (size_t i = 0; i < key_num; i++) {
    Key key = req_data.keys[i];
    T val = req_data.vals[i];
    if (init_weight_to_optim_[key]) {
      continue;
    } else {
      init_weight_to_optim_[key] = true;
    }
    ps_->InitWeightKeyToOptims(key, val);
  }
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleInitInputsShape(const ::ps::KVMeta &req_meta,
                                                              const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  const Key &key = req_data.keys[0];
  if (init_optim_info_[key]) {
    return;
  } else {
    init_optim_info_[key] = true;
  }
  ps_->InitOptimInputsShape(req_data.keys, req_data.vals, req_data.lens);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleInitEmbeddings(const ::ps::KVMeta &req_meta,
                                                             const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  const Key &key = req_data.keys[0];
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> shapes =
    std::make_shared<std::vector<std::shared_ptr<std::vector<size_t>>>>();
  std::shared_ptr<std::vector<size_t>> input_shape = std::make_shared<std::vector<size_t>>();
  std::shared_ptr<std::vector<size_t>> indices_shape = std::make_shared<std::vector<size_t>>();
  std::shared_ptr<std::vector<size_t>> output_shape = std::make_shared<std::vector<size_t>>();
  shapes->push_back(input_shape);
  shapes->push_back(indices_shape);
  shapes->push_back(output_shape);

  const Lengths &lens = req_data.lens;
  size_t index = 0;
  for (int i = 0; i < lens[0]; i++) {
    input_shape->push_back(static_cast<size_t>(req_data.vals[index++]));
  }
  for (int j = 0; j < lens[1]; j++) {
    indices_shape->push_back(static_cast<size_t>(req_data.vals[index++]));
  }
  for (int k = 0; k < lens[2]; k++) {
    output_shape->push_back(static_cast<size_t>(req_data.vals[index++]));
  }
  ps_->InitEmbeddingTable(key, shapes);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleCheckReadyForPush(const ::ps::KVMeta &req_meta,
                                                                const ::ps::KVPairs<T> &req_data,
                                                                ::ps::KVPairs<T> *res) {
  const Key &key = req_data.keys[0];
  bool ready = ps_->ReadyForPush(key);
  res->keys.push_back(key);
  res->vals.push_back(ready);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleCheckReadyForPull(const ::ps::KVMeta &req_meta,
                                                                const ::ps::KVPairs<T> &req_data,
                                                                ::ps::KVPairs<T> *res) {
  const Key &key = req_data.keys[0];
  bool ready = ps_->ReadyForPull(key);
  res->keys.push_back(key);
  res->vals.push_back(ready);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleEmbeddingLookup(const ::ps::KVMeta &req_meta,
                                                              const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res) {
  const Key &key = req_data.keys[0];
  for (size_t i = 0; i < req_data.keys.size(); i++) {
    res->keys.push_back(req_data.keys[i]);
  }
  ps_->DoEmbeddingLookup(key, req_data.keys.segment(1, req_data.keys.size()), res);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleFinalize(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                       ::ps::KVPairs<T> *res) {
  ps_->Finalize();
}

template <typename T>
bool ParameterServer<T>::Init(const FuncGraphPtr &func_graph) {
  pserver_num_ = ::ps::NumServers();
  worker_num_ = ::ps::NumWorkers();
  func_graph_ = func_graph;
  rank_id_ = ::ps::MyRank();
  handler_.reset(new ServerHandler(this));
  handler_->Init();

  InitOptimInfoBuilders();
  ps_->set_request_handle(*handler_);
  thread_.reset(new std::thread(&ParameterServer::UpdateWeights, this));
  return true;
}

template <typename T>
void ParameterServer<T>::InitOptimInfoBuilders() {
  std::shared_ptr<OptimizerInfoBuilder> momentum_info_builder = std::make_shared<MomentumOptimInfoBuilder>();
  std::shared_ptr<OptimizerInfoBuilder> sparse_adam_info_builder = std::make_shared<SparseAdamOptimInfoBuilder>();
  std::shared_ptr<OptimizerInfoBuilder> sparse_ftrl_info_builder = std::make_shared<SparseFtrlOptimInfoBuilder>();
  optim_info_builders_[kApplyMomentum] = momentum_info_builder;
  optim_info_builders_[kSparseAdam] = sparse_adam_info_builder;
  optim_info_builders_[kSparseFtrl] = sparse_ftrl_info_builder;
}

template <typename T>
void ParameterServer<T>::InitWeightKeyToOptims(const Key &key, const int &optim_id) {
  if (weight_key_to_optims_.count(key) > 0 || Util::optimizer_name(optim_id) == "") {
    return;
  }
  weight_key_to_optims_[key] = Util::optimizer_name(optim_id);
  weight_key_to_optim_op_[key] = Util::optimizer_node_name(optim_id);
}

template <typename T>
void ParameterServer<T>::InitOptimInputsShape(const Keys &keys, const Values &values, const Lengths &lengths) {
  InputsShapePtr inputs_shape = std::make_shared<InputsShape>();
  int val_idx = 0;
  const Key &key = keys[0];

  if (optim_inputs_shape_.count(key) == 0) {
    optim_inputs_shape_[key] = inputs_shape;
  }
  for (size_t i = 0; i < keys.size(); i++) {
    auto shape = std::make_shared<std::vector<size_t>>();
    inputs_shape->push_back(shape);

    int len = lengths[i];
    for (int j = 0; j < len; j++) {
      shape->push_back(values[val_idx++]);
    }
  }
  if (weight_key_to_optims_.count(key) > 0) {
    const std::string &optim_name = weight_key_to_optims_[key];
    const std::string &optim_op_name = weight_key_to_optim_op_[key];
    if (optimizers_.count(key) == 0 && optim_inputs_shape_.count(key) > 0) {
      const CNodePtr cnode = GetCNode(optim_op_name);
      MS_EXCEPTION_IF_NULL(cnode);
      if (optim_name == kSparseAdam) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyAdamPSKernel>(rank_id_, pserver_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseLazyAdam) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyLazyAdamPSKernel>(rank_id_, pserver_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kApplyMomentum) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::ApplyMomentumPSKernel>(rank_id_, pserver_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseFtrl) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyFtrlPSKernel>(rank_id_, pserver_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      }
    }
  }
}

template <typename T>
const CNodePtr ParameterServer<T>::GetCNode(const std::string &name) const {
  std::list<CNodePtr> cnodes = func_graph_->GetOrderedCnodes();
  for (CNodePtr cnode : cnodes) {
    std::string fullname = cnode->fullname_with_scope();
    if (fullname.find(name) != std::string::npos && fullname.find("Push") != std::string::npos) {
      return cnode;
    }
  }
  return nullptr;
}

template <typename T>
void ParameterServer<T>::InitWeight(const Key &key, const WeightPtr &weight) {
  MS_LOG(INFO) << "Initializing weight for key " << key;
  if ((weights_.count(key) == 0) || (is_embedding_[key] && weights_.count(key) != 0)) {
    weights_[key] = weight;
    tokens_[key] = 0;
    is_embedding_[key] = false;
  }
}

template <typename T>
void ParameterServer<T>::InitGrad(const Key &key, const GradPtr &grad) {
  if (grads_.count(key) == 0) {
    grads_[key] = grad;
    grads_accum_counter_[key] = 0;
  }
}

template <typename T>
void ParameterServer<T>::InitEmbeddingTable(
  const Key &key, const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes) {
  MS_LOG(INFO) << "Initializing embedding table for key " << key;
  std::shared_ptr<PServerKernel> lookup = std::make_shared<kernel::ps::EmbeddingLookUpPSKernel>(rank_id_, pserver_num_);
  lookup->InitKernel(shapes);
  embedding_lookup_ops_[key] = lookup;

  // Init embedding weight
  const std::vector<size_t> &input_shapes = lookup->input_sizes();
  size_t total_dims = 1;
  for (auto shape : input_shapes) {
    total_dims *= shape;
  }

  WeightPtr embedding = std::make_shared<Weight>(total_dims, 0);
  T *embedding_data = embedding->data();
  std::default_random_engine engine;
  std::normal_distribution<float> random(0, 0.01);
  for (size_t i = 0; i < total_dims; i++) {
    embedding_data[i] = random(engine);
  }
  weights_[key] = embedding;
  tokens_[key] = 0;
  is_embedding_[key] = true;

  grads_accum_counter_[key] = 0;
}

template <typename T>
void ParameterServer<T>::Finalize() {
  running_ = false;
  apply_grads_cv_.notify_one();
}

template <typename T>
void ParameterServer<T>::UpdateWeights() {
  while (true) {
    std::unique_lock<std::mutex> lock(mutex_);
    apply_grads_cv_.wait(lock, [this] { return this->ReadyForUpdateWeights() || !running_; });
    if (!running_) {
      break;
    }

    for (auto iter = weights_.begin(); iter != weights_.end(); iter++) {
      Key key = iter->first;
      WeightPtr weight_ptr = iter->second;

      std::shared_ptr<PServerKernel> optimizer = nullptr;
      if (weight_key_to_optims_.count(key) > 0) {
        optimizer = optimizers_[key];
      }
      MS_EXCEPTION_IF_NULL(optimizer);

      std::shared_ptr<OptimizerInfo> optim_info = optim_infos_[key];
      if (optim_info == nullptr) {
        continue;
      }
      const std::vector<kernel::AddressPtr> &inputs = optim_info->inputs();
      const std::vector<kernel::AddressPtr> &workspaces = optim_info->workspaces();
      const std::vector<kernel::AddressPtr> &outputs = optim_info->outputs();

      optim_info->ComputeMean(worker_num_);
      optimizer->Execute(inputs, workspaces, outputs);
      optim_info->Reset();
      if (!is_embedding_[key]) {
        tokens_[key] = worker_num_;
      }
    }
    ResetGradAccumCount();
  }
}

template <typename T>
void ParameterServer<T>::AccumGrad(const Keys &keys, const Values &values, const Lengths &lengths) {
  std::unique_lock<std::mutex> lock(mutex_);
  const Key &key = keys[0];
  std::shared_ptr<OptimizerInfo> optim_info = optim_infos_[key];

  // Create or update the optimizer info
  if (optim_info == nullptr) {
    const std::shared_ptr<OptimizerInfoBuilder> &builder = optim_info_builders_[weight_key_to_optims_[key]];
    std::shared_ptr<kernel::ps::PServerKernel> pserver_kernel = optimizers_[key];
    if (pserver_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "no optimizer found for key " << key << " optim name " << weight_key_to_optims_[key];
    }
    MS_EXCEPTION_IF_NULL(pserver_kernel);
    OptimizerInfo *optim =
      builder->Build(pserver_kernel, weights_[key], keys, values, lengths, optim_inputs_shape_[key], worker_num_);
    optim_info.reset(optim);
    optim_infos_[key] = optim_info;
  } else {
    optim_info->Update(values, lengths);
    optim_info->Accumulate(values, lengths);
  }

  grads_accum_counter_[key] += 1;
  if (grads_accum_counter_[key] == worker_num_) {
    grad_accum_count_++;
  }
  if (ReadyForUpdateWeights()) {
    apply_grads_cv_.notify_one();
  }
}

template <typename T>
WeightPtr ParameterServer<T>::weight(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (weights_.count(key) == 0) {
    MS_LOG(EXCEPTION) << "Invalid weight key " << key;
  }
  WeightPtr weight_ptr = weights_[key];
  WeightPtr copy_weight_ptr = std::make_shared<::ps::SArray<T>>(weight_ptr->size(), 0);
  copy_weight_ptr->CopyFrom(weight_ptr->data(), weight_ptr->size());
  tokens_[key] -= 1;
  return copy_weight_ptr;
}

template <typename T>
void ParameterServer<T>::DoEmbeddingLookup(Key key, const LookupIds &lookup_ids, ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (weights_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding table key " << key;
    return;
  }
  if (embedding_lookup_ops_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding lookup op key " << key;
    return;
  }
  WeightPtr table_ptr = weights_[key];
  std::shared_ptr<PServerKernel> table_lookup_op = embedding_lookup_ops_[key];

  // Update shapes of lookup operator
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> shapes =
    std::make_shared<std::vector<std::shared_ptr<std::vector<size_t>>>>();
  std::shared_ptr<std::vector<size_t>> indices_shape = std::make_shared<std::vector<size_t>>();
  indices_shape->emplace_back(lookup_ids.size());
  shapes->push_back(indices_shape);
  table_lookup_op->ReInit(shapes);

  const std::vector<size_t> output_shapes = table_lookup_op->output_sizes();
  std::vector<kernel::AddressPtr> inputs;
  AddressPtr embedding_table = std::make_shared<kernel::Address>();
  AddressPtr indices = std::make_shared<kernel::Address>();
  inputs.push_back(embedding_table);
  inputs.push_back(indices);
  embedding_table->addr = table_ptr->data();
  embedding_table->size = table_ptr->size() * sizeof(T);

  std::unique_ptr<int[]> tmp_ids(new int[lookup_ids.size()]);
  for (size_t i = 0; i < lookup_ids.size(); i++) {
    tmp_ids[i] = static_cast<int>(lookup_ids[i]);
  }
  indices->addr = tmp_ids.get();
  indices->size = lookup_ids.size() * sizeof(int);

  std::vector<kernel::AddressPtr> workspaces;
  std::vector<kernel::AddressPtr> outputs;
  AddressPtr output = std::make_shared<kernel::Address>();
  std::shared_ptr<Values> addr = std::make_shared<Values>(output_shapes[0] / sizeof(T), 0);

  output->addr = addr->data();
  output->size = output_shapes[0];
  outputs.push_back(output);

  table_lookup_op->Execute(inputs, workspaces, outputs);
  res->vals = *addr;
  res->lens.push_back(res->vals.size());
}

template <typename T>
int ParameterServer<T>::SumOfShapes(const std::vector<int> &shapes) const {
  int sum = 1;
  for (auto shape : shapes) {
    sum *= shape;
  }
  return sum;
}

template <typename T>
inline bool ParameterServer<T>::ReadyForUpdateWeights() {
  return grads_accum_counter_.size() > 0 && grad_accum_count_ == grads_accum_counter_.size();
}

template <typename T>
inline bool ParameterServer<T>::ReadyForPush(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (weights_.empty()) {
    MS_LOG(EXCEPTION) << "The weights in server is empty. Many reasons could cause this: 1.The Worker didn't send "
                         "kInitWeightsCmd command. 2.The Server failed to initialize weights.";
  }
  return grad_accum_count_ < weights_.size() && tokens_[key] <= 0;
}

template <typename T>
inline bool ParameterServer<T>::ReadyForPull(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (tokens_.count(key) == 0 || weights_[key] == 0) {
    MS_LOG(EXCEPTION) << "Invalid weight key " << key;
  }
  return tokens_[key] > 0;
}

template <typename T>
inline void ParameterServer<T>::ResetGradAccumCount() {
  grad_accum_count_ = 0;
  for (auto iter = grads_accum_counter_.begin(); iter != grads_accum_counter_.end(); iter++) {
    grads_accum_counter_[iter->first] = 0;
  }
}

template <typename T>
inline std::mutex &ParameterServer<T>::mutex() {
  return mutex_;
}

template <typename T>
void ParameterServer<T>::Run(const FuncGraphPtr &func_graph) {
  ::ps::Start(0);
  if (!::ps::IsServer()) {
    std::cout << "This is not ther Server" << std::endl;
    return;
  }
  Init(func_graph);
  thread_->join();
  ::ps::Finalize(0, true);
  exit(1);
}
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_PARAMETER_SERVER_H_
