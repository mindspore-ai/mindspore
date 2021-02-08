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
#include "ir/func_graph.h"
#include "backend/session/session_basic.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/session_factory.h"
#include "ps/common.h"
#include "ps/optimizer_info.h"
#include "ps/optimizer_info_builder.h"
#include "ps/util.h"
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

namespace mindspore {
namespace ps {
using mindspore::kernel::ps::PServerKernel;
using AnfAlgo = session::AnfRuntimeAlgorithm;
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
    ~ServerHandler() = default;
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
    void HandleUpdateEmbeddings(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);
    void HandleFinalize(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res);

    ParameterServer *ps_;
    typedef void (ServerHandler::*RequestHandler)(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                  ::ps::KVPairs<T> *res);
    std::unordered_map<int64_t, RequestHandler> handlers_;
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
  void DoEmbeddingLookup(Key key, const LookupIds &lookup_ids, ::ps::KVPairs<T> *res);
  void UpdateEmbeddings(const Key &key, const LookupIds &lookup_ids, const Values &vals);
  bool ReadyForUpdateWeights();
  bool ReadyForPush(const Key &key);
  bool ReadyForPull(const Key &key);
  void ResetGradAccumCount();
  const CNodePtr GetCNode(const std::string &name) const;
  std::mutex &mutex();
  void GetEmbeddingTableParamPtr();
  void SyncEmbeddingTables();

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
  std::map<Key, ParameterPtr> embedding_tables_;

  friend class ServerHandler;
};

class FuncGraph;
template <typename T>
void ParameterServer<T>::ServerHandler::operator()(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                   ::ps::KVServer<T> *server) {
  MS_EXCEPTION_IF_NULL(server);
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
  handlers_[kUpdateEmbeddingsCmd] = &ServerHandler::HandleUpdateEmbeddings;
  handlers_[kFinalizeCmd] = &ServerHandler::HandleFinalize;
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandlePushReq(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                      ::ps::KVPairs<T> *res) {
  MS_EXCEPTION_IF_NULL(res);
  ps_->AccumGrad(req_data.keys, req_data.vals, req_data.lens);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandlePullReq(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                      ::ps::KVPairs<T> *res) {
  MS_EXCEPTION_IF_NULL(res);
  res->keys = req_data.keys;
  ::ps::Key key = req_data.keys[0];
  res->vals = *(ps_->weight(key));
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleInitWeights(const ::ps::KVMeta &req_meta,
                                                          const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(res);
  size_t key_num = req_data.keys.size();
  T *data_ptr = req_data.vals.data();
  size_t pos = 0;
  for (size_t i = 0; i < key_num; i++) {
    Key key = req_data.keys[i];
    size_t data_len = req_data.lens.size() != key_num ? req_data.vals.size() / key_num : req_data.lens[i];

    if (!ps_->HasWeight(key)) {
      WeightPtr weight_ptr = std::make_shared<::ps::SArray<T>>();
      MS_EXCEPTION_IF_NULL(weight_ptr);
      weight_ptr->CopyFrom(data_ptr + pos, data_len);
      ps_->InitWeight(key, weight_ptr);

      GradPtr grad_ptr = std::make_shared<::ps::SArray<T>>(data_len, 0);
      MS_EXCEPTION_IF_NULL(grad_ptr);
      ps_->InitGrad(key, grad_ptr);
    }
    pos += data_len;
  }
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleInitWeightToOptimId(const ::ps::KVMeta &req_meta,
                                                                  const ::ps::KVPairs<T> &req_data,
                                                                  ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(res);
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
  MS_EXCEPTION_IF_NULL(res);
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
  MS_EXCEPTION_IF_NULL(res);
  const Key &key = req_data.keys[0];
  MS_LOG(INFO) << "Initializing embedding table for key:" << key;
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> shapes =
    std::make_shared<std::vector<std::shared_ptr<std::vector<size_t>>>>();
  MS_EXCEPTION_IF_NULL(shapes);
  std::shared_ptr<std::vector<size_t>> input_shape = std::make_shared<std::vector<size_t>>();
  MS_EXCEPTION_IF_NULL(input_shape);
  std::shared_ptr<std::vector<size_t>> indices_shape = std::make_shared<std::vector<size_t>>();
  MS_EXCEPTION_IF_NULL(indices_shape);
  std::shared_ptr<std::vector<size_t>> output_shape = std::make_shared<std::vector<size_t>>();
  MS_EXCEPTION_IF_NULL(output_shape);
  shapes->push_back(input_shape);
  shapes->push_back(indices_shape);
  shapes->push_back(output_shape);

  const Lengths &lens = req_data.lens;
  size_t index = 0;
  for (int64_t i = 0; i < lens[0]; i++) {
    input_shape->push_back(static_cast<size_t>(req_data.vals[index++]));
  }
  for (int64_t j = 0; j < lens[1]; j++) {
    indices_shape->push_back(static_cast<size_t>(req_data.vals[index++]));
  }
  for (int64_t k = 0; k < lens[2]; k++) {
    output_shape->push_back(static_cast<size_t>(req_data.vals[index++]));
  }
  ParamInitInfo param_init_info;
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    param_init_info.param_type_ = static_cast<ParamType>(lens[3]);
    if (param_init_info.param_type_ == kWeight) {
      param_init_info.global_seed_ = static_cast<size_t>(lens[4]);
      param_init_info.op_seed_ = static_cast<size_t>(lens[5]);
    } else if (param_init_info.param_type_ == kAccumulation) {
      param_init_info.init_val_ = req_data.vals[index];
    }
  }
  ps_->InitEmbeddingTable(key, shapes, param_init_info);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleCheckReadyForPush(const ::ps::KVMeta &req_meta,
                                                                const ::ps::KVPairs<T> &req_data,
                                                                ::ps::KVPairs<T> *res) {
  MS_EXCEPTION_IF_NULL(res);
  const Key &key = req_data.keys[0];
  bool ready = ps_->ReadyForPush(key);
  res->keys.push_back(key);
  res->vals.push_back(ready);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleCheckReadyForPull(const ::ps::KVMeta &req_meta,
                                                                const ::ps::KVPairs<T> &req_data,
                                                                ::ps::KVPairs<T> *res) {
  MS_EXCEPTION_IF_NULL(res);
  const Key &key = req_data.keys[0];
  bool ready = ps_->ReadyForPull(key);
  res->keys.push_back(key);
  res->vals.push_back(ready);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleEmbeddingLookup(const ::ps::KVMeta &req_meta,
                                                              const ::ps::KVPairs<T> &req_data, ::ps::KVPairs<T> *res) {
  MS_EXCEPTION_IF_NULL(res);
  const Key &key = req_data.keys[0];
  for (size_t i = 1; i < req_data.keys.size(); i++) {
    res->keys.push_back(req_data.keys[i]);
  }
  ps_->DoEmbeddingLookup(key, req_data.keys.segment(1, req_data.keys.size()), res);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleUpdateEmbeddings(const ::ps::KVMeta &req_meta,
                                                               const ::ps::KVPairs<T> &req_data,
                                                               ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(res);
  const Key &key = req_data.keys[0];
  const LookupIds &lookup_ids = req_data.keys.segment(1, req_data.keys.size());
  const Values &update_vals = req_data.vals;
  ps_->UpdateEmbeddings(key, lookup_ids, update_vals);
}

template <typename T>
void ParameterServer<T>::ServerHandler::HandleFinalize(const ::ps::KVMeta &req_meta, const ::ps::KVPairs<T> &req_data,
                                                       ::ps::KVPairs<T> *res) {
  MS_EXCEPTION_IF_NULL(res);
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
  GetEmbeddingTableParamPtr();
  return true;
}

template <typename T>
void ParameterServer<T>::InitOptimInfoBuilders() {
  std::shared_ptr<OptimizerInfoBuilder> momentum_info_builder = std::make_shared<MomentumOptimInfoBuilder>(worker_num_);
  std::shared_ptr<OptimizerInfoBuilder> sparse_adam_info_builder =
    std::make_shared<SparseAdamOptimInfoBuilder>(worker_num_);
  std::shared_ptr<OptimizerInfoBuilder> sparse_ftrl_info_builder =
    std::make_shared<SparseFtrlOptimInfoBuilder>(worker_num_);
  optim_info_builders_[kApplyMomentum] = momentum_info_builder;
  optim_info_builders_[kSparseAdam] = sparse_adam_info_builder;
  optim_info_builders_[kSparseFtrl] = sparse_ftrl_info_builder;
}

template <typename T>
void ParameterServer<T>::InitWeightKeyToOptims(const Key &key, const int64_t &optim_id) {
  if (weight_key_to_optims_.count(key) > 0 || Util::optimizer_name(optim_id) == "") {
    return;
  }
  weight_key_to_optims_[key] = Util::optimizer_name(optim_id);
  weight_key_to_optim_op_[key] = Util::optimizer_node_name(optim_id);
  MS_LOG(INFO) << "Initializing optimizer id for key:" << key << ", optimizer name:" << weight_key_to_optims_[key]
               << ", optimizer op name:" << weight_key_to_optim_op_[key];
}

template <typename T>
void ParameterServer<T>::InitOptimInputsShape(const Keys &keys, const Values &values, const Lengths &lengths) {
  InputsShapePtr inputs_shape = std::make_shared<InputsShape>();
  MS_EXCEPTION_IF_NULL(inputs_shape);
  InputsShapePtr original_inputs_shape = std::make_shared<InputsShape>();
  MS_EXCEPTION_IF_NULL(original_inputs_shape);
  int64_t val_idx = 0;
  const Key &key = keys[0];
  MS_LOG(INFO) << "Initializing optimizer inputs shape for key:" << key;
  if (optim_inputs_shape_.count(key) == 0) {
    original_optim_inputs_shape_[key] = original_inputs_shape;
    optim_inputs_shape_[key] = inputs_shape;
  }
  for (size_t i = 0; i < keys.size(); i++) {
    auto shape = std::make_shared<std::vector<size_t>>();
    MS_EXCEPTION_IF_NULL(shape);
    auto original_shape = std::make_shared<std::vector<size_t>>();
    MS_EXCEPTION_IF_NULL(original_shape);
    inputs_shape->push_back(shape);
    original_inputs_shape->push_back(original_shape);

    for (int64_t j = 0; j < lengths[i]; j++) {
      shape->push_back(values[val_idx]);
      original_shape->push_back(values[val_idx++]);
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
          std::make_shared<kernel::ps::SparseApplyAdamPSKernel>(rank_id_, pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseLazyAdam) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyLazyAdamPSKernel>(rank_id_, pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kApplyMomentum) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::ApplyMomentumPSKernel>(rank_id_, pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseFtrl) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyFtrlPSKernel>(rank_id_, pserver_num_, worker_num_);
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
    MS_EXCEPTION_IF_NULL(cnode);
    std::string fullname = cnode->fullname_with_scope();
    if (fullname.find(name) != std::string::npos && fullname.find("Push") != std::string::npos) {
      return cnode;
    }
  }
  return nullptr;
}

template <typename T>
void ParameterServer<T>::InitWeight(const Key &key, const WeightPtr &weight) {
  MS_EXCEPTION_IF_NULL(weight);
  if ((weights_.count(key) == 0) || (is_embedding_[key] && weights_.count(key) != 0)) {
    MS_LOG(INFO) << "Initializing weight for key " << key << ", server rank " << rank_id_;
    weights_[key] = weight;
    tokens_[key] = 0;
    is_embedding_[key] = false;
  }
}

template <typename T>
void ParameterServer<T>::InitGrad(const Key &key, const GradPtr &grad) {
  MS_EXCEPTION_IF_NULL(grad);
  if (grads_.count(key) == 0) {
    grads_[key] = grad;
    grads_accum_counter_[key] = 0;
  }
}

template <typename T>
void ParameterServer<T>::InitEmbeddingTable(
  const Key &key, const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes,
  const ParamInitInfo &param_init_info) {
  MS_EXCEPTION_IF_NULL(shapes);
  if (weights_.count(key) == 0) {
    std::shared_ptr<PServerKernel> lookup =
      std::make_shared<kernel::ps::EmbeddingLookUpPSKernel>(rank_id_, pserver_num_, worker_num_);
    lookup->InitKernel(shapes);
    embedding_lookup_ops_[key] = lookup;

    // Init embedding weight
    const std::vector<size_t> &input_shapes = lookup->input_sizes();
    size_t total_dims =
      std::accumulate(input_shapes.begin(), input_shapes.end(), IntToSize(1), std::multiplies<size_t>());
    WeightPtr embedding = std::make_shared<Weight>(total_dims, 0);
    MS_EXCEPTION_IF_NULL(embedding);
    T *embedding_data = embedding->data();
    std::default_random_engine engine;
    std::normal_distribution<float> random(0, 0.01);
    if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
      if (param_init_info.param_type_ == kWeight) {
        InitRandomNormal(0, 0.01, input_shapes, param_init_info.global_seed_, param_init_info.op_seed_, embedding_data);
      } else if (param_init_info.param_type_ == kAccumulation) {
        for (size_t i = 0; i < total_dims; i++) {
          embedding_data[i] = param_init_info.init_val_;
        }
      }
    } else {
      for (size_t i = 0; i < total_dims; i++) {
        embedding_data[i] = random(engine);
      }
    }
    weights_[key] = embedding;
    tokens_[key] = 0;
    is_embedding_[key] = true;

    grads_accum_counter_[key] = 0;
  }
}

template <typename T>
bool ParameterServer<T>::HasWeight(const Key &key) {
  return (weights_.count(key) > 0 && !is_embedding_.count(key));
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
      if (optim_info != nullptr) {
        const std::vector<kernel::AddressPtr> &inputs = optim_info->inputs();
        const std::vector<kernel::AddressPtr> &workspaces = optim_info->workspaces();
        const std::vector<kernel::AddressPtr> &outputs = optim_info->outputs();

        std::vector<std::vector<size_t>> shapes = {};
        std::vector<size_t> indices_shape = {};
        indices_shape.emplace_back(optim_info->indice_size());
        shapes.push_back(indices_shape);

        if (original_optim_inputs_shape_.count(key) != 0) {
          for (auto input_shapes : *(original_optim_inputs_shape_[key])) {
            shapes.push_back(*input_shapes);
          }
        }
        optimizer->ReInit(shapes);
        optim_info->ComputeMean(shapes, worker_num_, pserver_num_, rank_id_);
        optimizer->Execute(inputs, workspaces, outputs);
        optim_info->Reset();
      }
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
  bool no_sparse_grad = values.size() == 1 && values[0] == -100;
  if (!no_sparse_grad) {
    std::shared_ptr<OptimizerInfo> optim_info = optim_infos_[key];

    // Create or update the optimizer info
    if (optim_info == nullptr) {
      const std::shared_ptr<OptimizerInfoBuilder> &builder = optim_info_builders_[weight_key_to_optims_[key]];
      std::shared_ptr<kernel::ps::PServerKernel> pserver_kernel = optimizers_[key];
      if (pserver_kernel == nullptr) {
        MS_LOG(EXCEPTION) << "no optimizer found for key " << key << " optim name " << weight_key_to_optims_[key];
      }
      MS_EXCEPTION_IF_NULL(pserver_kernel);
      OptimizerInfo *optim = builder->Build(pserver_kernel, weights_[key], keys, values, lengths,
                                            optim_inputs_shape_[key], worker_num_, is_embedding_[key]);
      optim_info.reset(optim);
      optim_infos_[key] = optim_info;
    } else {
      optim_info->Update(values, lengths);
      optim_info->Accumulate(values, lengths);
    }
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
  MS_EXCEPTION_IF_NULL(weight_ptr);
  WeightPtr copy_weight_ptr = std::make_shared<::ps::SArray<T>>(weight_ptr->size(), 0);
  MS_EXCEPTION_IF_NULL(copy_weight_ptr);
  copy_weight_ptr->CopyFrom(weight_ptr->data(), weight_ptr->size());
  tokens_[key] -= 1;
  return copy_weight_ptr;
}

template <typename T>
void ParameterServer<T>::DoEmbeddingLookup(Key key, const LookupIds &lookup_ids, ::ps::KVPairs<T> *res) {
  std::unique_lock<std::mutex> lock(mutex_);
  MS_EXCEPTION_IF_NULL(res);
  if (weights_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding table key " << key;
    return;
  }
  if (embedding_lookup_ops_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding lookup op key " << key;
    return;
  }
  WeightPtr table_ptr = weights_[key];
  MS_EXCEPTION_IF_NULL(table_ptr);
  std::shared_ptr<PServerKernel> table_lookup_op = embedding_lookup_ops_[key];
  MS_EXCEPTION_IF_NULL(table_lookup_op);

  // Update shapes of lookup operator
  std::vector<std::vector<size_t>> shapes = {};
  std::vector<size_t> indices_shape = {};
  indices_shape.emplace_back(lookup_ids.size());
  shapes.push_back(indices_shape);
  table_lookup_op->ReInit(shapes);

  const std::vector<size_t> output_shapes = table_lookup_op->output_sizes();
  std::vector<kernel::AddressPtr> inputs;
  AddressPtr embedding_table = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(embedding_table);
  AddressPtr indices = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(indices);
  inputs.push_back(embedding_table);
  inputs.push_back(indices);
  embedding_table->addr = table_ptr->data();
  embedding_table->size = table_ptr->size() * sizeof(T);

  std::unique_ptr<int[]> tmp_ids(new int[lookup_ids.size()]);
  MS_EXCEPTION_IF_NULL(tmp_ids);
  for (size_t i = 0; i < lookup_ids.size(); i++) {
    tmp_ids[i] = static_cast<int>(lookup_ids[i]);
  }
  indices->addr = tmp_ids.get();
  indices->size = lookup_ids.size() * sizeof(int);

  std::vector<kernel::AddressPtr> workspaces;
  std::vector<kernel::AddressPtr> outputs;
  AddressPtr output = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(output);
  std::shared_ptr<Values> addr = std::make_shared<Values>(output_shapes[0] / sizeof(T), 0);
  MS_EXCEPTION_IF_NULL(addr);

  output->addr = addr->data();
  output->size = output_shapes[0];
  outputs.push_back(output);

  table_lookup_op->Execute(inputs, workspaces, outputs);
  res->vals = *addr;
  res->lens.push_back(res->vals.size());
}

template <typename T>
void ParameterServer<T>::UpdateEmbeddings(const Key &key, const LookupIds &lookup_ids, const Values &vals) {
  if (weights_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding table key " << key;
    return;
  }
  if (embedding_lookup_ops_.count(key) == 0) {
    MS_LOG(ERROR) << "Invalid embedding lookup op key " << key;
    return;
  }
  WeightPtr table_ptr = weights_[key];
  MS_EXCEPTION_IF_NULL(table_ptr);
  std::shared_ptr<PServerKernel> table_lookup_op = embedding_lookup_ops_[key];
  MS_EXCEPTION_IF_NULL(table_lookup_op);
  table_lookup_op->UpdateEmbeddings(table_ptr->data(), lookup_ids.data(), vals.data(), lookup_ids.size());
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
void ParameterServer<T>::GetEmbeddingTableParamPtr() {
  MS_EXCEPTION_IF_NULL(func_graph_);
  auto cnodes = func_graph_->GetOrderedCnodes();
  Key count = 0;
  for (auto cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    std::string cnode_name = AnfAlgo::GetCNodeName(cnode);
    if (cnode_name == kEmbeddingLookupOpName || cnode_name == kGatherV2OpName || cnode_name == kSparseGatherV2OpName) {
      auto embedding_table = AnfAlgo::GetInputNode(cnode, 0);
      if (IsPrimitiveCNode(embedding_table, prim::kPrimLoad)) {
        auto embedding_cnode = embedding_table->cast<CNodePtr>();
        embedding_table = AnfAlgo::GetInputNode(embedding_cnode, 0);
      }
      MS_EXCEPTION_IF_NULL(embedding_table);
      if (embedding_table->isa<Parameter>()) {
        MS_LOG(INFO) << "Embedding table name is " << embedding_table->fullname_with_scope() << ", key is " << count;
        embedding_tables_.insert(std::make_pair(count, embedding_table->cast<ParameterPtr>()));
        count++;
      }
    }
  }
}

template <typename T>
void ParameterServer<T>::SyncEmbeddingTables() {
  for (auto embedding_table : embedding_tables_) {
    Key key = embedding_table.first;
    if (embedding_lookup_ops_.count(key) == 0) {
      MS_LOG(WARNING) << "Can't find look up PS kernel for key " << key;
      continue;
    }
    auto lookup = embedding_lookup_ops_[key];
    const std::vector<size_t> &input_shapes = lookup->input_sizes();
    std::vector<int64_t> new_tensor_shape(input_shapes.begin(), input_shapes.end());

    tensor::TensorPtr new_tensor = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, new_tensor_shape);
    MS_EXCEPTION_IF_NULL(new_tensor);
    float *new_tensor_data_ptr = reinterpret_cast<float *>(new_tensor->data_c());
    size_t new_tensor_size = static_cast<size_t>(new_tensor->data().nbytes());
    size_t embedding_table_size = weights_[key]->size() * sizeof(float);
    if (new_tensor_size != embedding_table_size) {
      MS_LOG(EXCEPTION) << "Shape of embedding table can't match. New tensor size:" << new_tensor_size
                        << ", embedding_table size:" << embedding_table_size;
    }
    MS_EXCEPTION_IF_NULL(new_tensor_data_ptr);
    MS_EXCEPTION_IF_NULL(weights_[key]->data());
    int64_t ret = memcpy_s(new_tensor_data_ptr, new_tensor_size, weights_[key]->data(), embedding_table_size);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
      return;
    }

    auto paramter_tensor_ptr = embedding_table.second->default_param();
    MS_EXCEPTION_IF_NULL(paramter_tensor_ptr);
    paramter_tensor_ptr->cast<tensor::TensorPtr>()->AssignValue(*new_tensor);
  }
}

template <typename T>
void ParameterServer<T>::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "PServer starts connecting to scheduler and workers...";
  ::ps::Start(0);
  MS_LOG(INFO) << "PServer connected successfully.";
  if (!::ps::IsServer()) {
    std::cout << "This is not ther Server" << std::endl;
    return;
  }
  Init(func_graph);
  PSContext::instance()->SetPSRankId(rank_id_);
  thread_->join();
  SyncEmbeddingTables();
  MS_LOG(INFO) << "PServer finished updating models, starts finalizing...";
  ::ps::Finalize(0, true);
  MS_LOG(INFO) << "PServer finalized successfully.";
}
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PARAMETER_SERVER_H_
