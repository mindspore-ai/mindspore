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

#include "ps/parameter_server.h"

namespace mindspore {
namespace ps {
void ParameterServer::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "PServer starts connecting to scheduler and workers...";
  server_node_ = std::make_shared<core::ServerNode>();

  MS_LOG(INFO) << "PServer connected successfully.";
  if (!PSContext::instance()->is_server()) {
    MS_LOG(INFO) << "This is not the Server node.";
    return;
  }
  Init(func_graph);
  server_node_->Start();
  PSContext::instance()->SetPSRankId(server_node_->rank_id());
  thread_->join();
  SyncEmbeddingTables();
  MS_LOG(INFO) << "PServer finished updating models, starts finalizing...";
  server_node_->Finish();
  server_node_->Stop();
  MS_LOG(INFO) << "PServer finalized successfully.";
}

bool ParameterServer::Init(const FuncGraphPtr &func_graph) {
  pserver_num_ = std::strtol(mindspore::common::GetEnv(kEnvPServerNum).c_str(), nullptr, 10);
  worker_num_ = std::strtol(mindspore::common::GetEnv(kEnvWorkerNum).c_str(), nullptr, 10);
  func_graph_ = func_graph;
  handler_.reset(new ServerHandler(this));
  handler_->Init();

  InitOptimInfoBuilders();
  server_node_->set_handler(*handler_);
  server_node_->set_event_callback([&](const core::NodeEvent &event) {
    if ((event == core::NodeEvent::CLUSTER_TIMEOUT) ||
        (event == core::NodeEvent::SCHEDULER_TIMEOUT || (event == core::NodeEvent::NODE_TIMEOUT))) {
      MS_LOG(ERROR) << "Trigger timeout event:" << event << " begin to exit the system!";
      Finalize();
    }
  });
  thread_.reset(new std::thread(&ParameterServer::UpdateWeights, this));
  GetEmbeddingTableParamPtr();
  return true;
}

void ParameterServer::InitOptimInfoBuilders() {
  std::shared_ptr<OptimizerInfoBuilder> momentum_info_builder = std::make_shared<MomentumOptimInfoBuilder>(worker_num_);
  std::shared_ptr<OptimizerInfoBuilder> sparse_adam_info_builder =
    std::make_shared<SparseAdamOptimInfoBuilder>(worker_num_);
  std::shared_ptr<OptimizerInfoBuilder> sparse_ftrl_info_builder =
    std::make_shared<SparseFtrlOptimInfoBuilder>(worker_num_);
  optim_info_builders_[kApplyMomentum] = momentum_info_builder;
  optim_info_builders_[kSparseAdam] = sparse_adam_info_builder;
  optim_info_builders_[kSparseFtrl] = sparse_ftrl_info_builder;
}

void ParameterServer::InitWeightKeyToOptims(const Key &key, const int64_t &optim_id) {
  if (weight_key_to_optims_.count(key) > 0 || Util::optimizer_name(optim_id) == "") {
    return;
  }
  weight_key_to_optims_[key] = Util::optimizer_name(optim_id);
  weight_key_to_optim_op_[key] = Util::optimizer_node_name(optim_id);
  MS_LOG(INFO) << "Initializing optimizer id for key:" << key << ", optimizer name:" << weight_key_to_optims_[key]
               << ", optimizer op name:" << weight_key_to_optim_op_[key];
}

void ParameterServer::InitOptimInputsShape(const Keys &keys, const Values &values, const Lengths &lengths) {
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
          std::make_shared<kernel::ps::SparseApplyAdamPSKernel>(server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseLazyAdam) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyLazyAdamPSKernel>(server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kApplyMomentum) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::ApplyMomentumPSKernel>(server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      } else if (optim_name == kSparseFtrl) {
        std::shared_ptr<PServerKernel> optimizer =
          std::make_shared<kernel::ps::SparseApplyFtrlPSKernel>(server_node_->rank_id(), pserver_num_, worker_num_);
        optimizer->InitKernel(cnode, optim_inputs_shape_[key]);
        optimizers_[key] = optimizer;
      }
    }
  }
}

void ParameterServer::InitWeight(const Key &key, const WeightPtr &weight) {
  MS_EXCEPTION_IF_NULL(weight);
  if ((weights_.count(key) == 0) || (is_embedding_[key] && weights_.count(key) != 0)) {
    MS_LOG(INFO) << "Initializing weight for key " << key << ", server rank " << server_node_->rank_id();
    weights_[key] = weight;
    tokens_[key] = 0;
    is_embedding_[key] = false;
  }
}

void ParameterServer::InitGrad(const Key &key, const GradPtr &grad) {
  MS_EXCEPTION_IF_NULL(grad);
  if (grads_.count(key) == 0) {
    grads_[key] = grad;
    grads_accum_counter_[key] = 0;
  }
}

void ParameterServer::InitEmbeddingTable(
  const Key &key, const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes,
  const ParamInitInfo &param_init_info) {
  MS_EXCEPTION_IF_NULL(shapes);
  if (weights_.count(key) == 0) {
    std::shared_ptr<PServerKernel> lookup =
      std::make_shared<kernel::ps::EmbeddingLookUpPSKernel>(server_node_->rank_id(), pserver_num_, worker_num_);
    lookup->InitKernel(shapes);
    embedding_lookup_ops_[key] = lookup;

    // Init embedding weight
    const std::vector<size_t> &input_shapes = lookup->input_sizes();
    size_t total_dims =
      std::accumulate(input_shapes.begin(), input_shapes.end(), IntToSize(1), std::multiplies<size_t>());
    WeightPtr embedding = std::make_shared<Weight>(total_dims, 0);
    MS_EXCEPTION_IF_NULL(embedding);
    float *embedding_data = embedding->data();
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
    MS_LOG(DEBUG) << "The key:" << key << " the embedding:" << *embedding;
    tokens_[key] = 0;
    is_embedding_[key] = true;

    grads_accum_counter_[key] = 0;
  }
}

bool ParameterServer::HasWeight(const Key &key) { return (weights_.count(key) > 0 && !is_embedding_.count(key)); }

void ParameterServer::Finalize() {
  running_ = false;
  apply_grads_cv_.notify_one();
}

void ParameterServer::UpdateWeights() {
  while (true) {
    MS_LOG(INFO) << "The running is:" << running_ << " the ready is:" << this->ReadyForUpdateWeights();
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
          std::transform(
            (*(original_optim_inputs_shape_[key])).begin(), (*(original_optim_inputs_shape_[key])).end(),
            std::back_inserter(shapes),
            [](std::shared_ptr<std::vector<size_t>> input_shapes) -> std::vector<size_t> { return *input_shapes; });
        }
        optimizer->ReInit(shapes);
        optim_info->ComputeMean(shapes, worker_num_, pserver_num_, server_node_->rank_id());
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

void ParameterServer::AccumGrad(const Keys &keys, const Values &values, const Lengths &lengths) {
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

WeightPtr ParameterServer::weight(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (weights_.count(key) == 0) {
    MS_LOG(EXCEPTION) << "Invalid weight key " << key;
  }
  WeightPtr weight_ptr = weights_[key];
  MS_EXCEPTION_IF_NULL(weight_ptr);
  WeightPtr copy_weight_ptr = std::make_shared<std::vector<float>>(weight_ptr->size(), 0);
  MS_EXCEPTION_IF_NULL(copy_weight_ptr);
  copy_weight_ptr = weight_ptr;
  tokens_[key] -= 1;
  return copy_weight_ptr;
}

void ParameterServer::DoEmbeddingLookup(Key key, const LookupIds &lookup_ids, KVMessage *res) {
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
  embedding_table->size = table_ptr->size() * sizeof(float);

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
  std::shared_ptr<Values> addr = std::make_shared<Values>(output_shapes[0] / sizeof(float), 0);
  MS_EXCEPTION_IF_NULL(addr);

  output->addr = addr->data();
  output->size = output_shapes[0];
  outputs.push_back(output);

  table_lookup_op->Execute(inputs, workspaces, outputs);
  *res->mutable_values() = {addr->begin(), addr->end()};
  res->add_len(res->values_size());
}

void ParameterServer::UpdateEmbeddings(const Key &key, const LookupIds &lookup_ids, const Values &vals) {
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

inline bool ParameterServer::ReadyForUpdateWeights() {
  return grads_accum_counter_.size() > 0 && grad_accum_count_ == grads_accum_counter_.size();
}

inline bool ParameterServer::ReadyForPush(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (weights_.empty()) {
    MS_LOG(EXCEPTION) << "The weights in server is empty. Many reasons could cause this: 1.The Worker didn't send "
                         "kInitWeightsCmd command. 2.The Server failed to initialize weights.";
  }
  MS_LOG(INFO) << "The grad_accum_count_:" << grad_accum_count_ << " the weights_:" << weights_.size()
               << " the token:" << (tokens_[key] <= 0);
  return grad_accum_count_ < weights_.size() && tokens_[key] <= 0;
}

inline bool ParameterServer::ReadyForPull(const Key &key) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (tokens_.count(key) == 0 || weights_[key] == 0) {
    MS_LOG(EXCEPTION) << "Invalid weight key " << key;
  }
  MS_LOG(INFO) << "ReadyForPull: " << (tokens_[key] > 0);
  return tokens_[key] > 0;
}

inline void ParameterServer::ResetGradAccumCount() {
  grad_accum_count_ = 0;
  for (auto iter = grads_accum_counter_.begin(); iter != grads_accum_counter_.end(); iter++) {
    grads_accum_counter_[iter->first] = 0;
  }
}

const CNodePtr ParameterServer::GetCNode(const std::string &name) const {
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

inline std::mutex &ParameterServer::mutex() { return mutex_; }

void ParameterServer::GetEmbeddingTableParamPtr() {
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

void ParameterServer::SyncEmbeddingTables() {
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

void ParameterServer::ServerHandler::Init() {
  handlers_[kInitWeightsCmd] = &ServerHandler::HandleInitWeights;
  handlers_[kInitWeightToOptimIdCmd] = &ServerHandler::HandleInitWeightToOptimId;
  handlers_[kInitOptimInputsShapeCmd] = &ServerHandler::HandleInitInputsShape;
  handlers_[kInitEmbeddingsCmd] = &ServerHandler::HandleInitEmbeddings;
  handlers_[kCheckReadyForPushCmd] = &ServerHandler::HandleCheckReadyForPush;
  handlers_[kCheckReadyForPullCmd] = &ServerHandler::HandleCheckReadyForPull;
  handlers_[kEmbeddingLookupCmd] = &ServerHandler::HandleEmbeddingLookup;
  handlers_[kUpdateEmbeddingsCmd] = &ServerHandler::HandleUpdateEmbeddings;
  handlers_[kFinalizeCmd] = &ServerHandler::HandleFinalize;
  handlers_[kPushCmd] = &ServerHandler::HandlePushReq;
  handlers_[kPullCmd] = &ServerHandler::HandlePullReq;
  commands_[kInitWeightsCmd] = "kInitWeightsCmd";
  commands_[kInitWeightToOptimIdCmd] = "kInitWeightToOptimIdCmd";
  commands_[kInitOptimInputsShapeCmd] = "kInitOptimInputsShapeCmd";
  commands_[kInitEmbeddingsCmd] = "kInitEmbeddingsCmd";
  commands_[kCheckReadyForPushCmd] = "kCheckReadyForPushCmd";
  commands_[kCheckReadyForPullCmd] = "kCheckReadyForPullCmd";
  commands_[kEmbeddingLookupCmd] = "kEmbeddingLookupCmd";
  commands_[kUpdateEmbeddingsCmd] = "kUpdateEmbeddingsCmd";
  commands_[kFinalizeCmd] = "kFinalizeCmd";
  commands_[kPushCmd] = "kPushCmd";
  commands_[kPullCmd] = "kPullCmd";
}

void ParameterServer::ServerHandler::operator()(std::shared_ptr<core::TcpConnection> conn,
                                                std::shared_ptr<core::MessageMeta> meta, DataPtr data, size_t size) {
  auto output = std::make_shared<std::vector<unsigned char>>();
  if (commands_.count(meta->user_cmd()) == 0) {
    MS_LOG(EXCEPTION) << "The command:" << meta->user_cmd() << " is not supported!";
  }
  MS_LOG(INFO) << "The command is:" << commands_[meta->user_cmd()];

  auto &handler_ptr = handlers_[meta->user_cmd()];
  (this->*handler_ptr)(data, size, output);
  MS_LOG(DEBUG) << "The output size is:" << output->size();

  if (output->size() > 0) {
    ps_->server_node_->Response(conn, meta, output->data(), output->size());
  } else {
    // If the size of the output is 0, then constructed an empty string, Because the Response function is a synchronous,
    // the res variable  will be automatically recycled after calling the Response function
    std::string res;
    ps_->server_node_->Response(conn, meta, res.data(), res.length());
  }
  MS_LOG(DEBUG) << "The request id is:" << meta->request_id() << " the current time is:"
                << std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now())
                     .time_since_epoch()
                     .count();
}

void ParameterServer::ServerHandler::HandlePushReq(DataPtr data, size_t size, VectorPtr res) {
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  Keys keys = {input.keys().begin(), input.keys().end()};
  Values values = {input.values().begin(), input.values().end()};
  Lengths lens = {input.len().begin(), input.len().end()};
  MS_LOG(DEBUG) << "The keys:" << keys << " the values:" << values << " the len:" << lens;
  ps_->AccumGrad(keys, values, lens);
}

void ParameterServer::ServerHandler::HandlePullReq(DataPtr data, size_t size, VectorPtr res) {
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  KVMessage res_data;
  *res_data.mutable_keys() = input.keys();
  Key key = input.keys()[0];
  auto weight = ps_->weight(key);
  *res_data.mutable_values() = {weight->begin(), weight->end()};
  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleInitWeights(DataPtr data, size_t size, VectorPtr res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  int key_num = input.keys_size();
  const float *data_ptr = input.values().data();
  size_t pos = 0;
  for (int i = 0; i < key_num; i++) {
    Key key = input.keys()[i];
    size_t data_len = input.len_size() != key_num ? input.values_size() / key_num : input.len()[i];

    if (!ps_->HasWeight(key)) {
      WeightPtr weight_ptr = std::make_shared<std::vector<float>>(data_ptr + pos, data_ptr + (pos + data_len));
      MS_EXCEPTION_IF_NULL(weight_ptr);
      ps_->InitWeight(key, weight_ptr);

      GradPtr grad_ptr = std::make_shared<std::vector<float>>(data_len, 0);
      MS_EXCEPTION_IF_NULL(grad_ptr);
      ps_->InitGrad(key, grad_ptr);
    }
    pos += data_len;
  }
}

void ParameterServer::ServerHandler::HandleInitWeightToOptimId(DataPtr data, size_t size, VectorPtr res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  size_t key_num = input.keys_size();
  for (size_t i = 0; i < key_num; i++) {
    Key key = input.keys()[i];
    float val = input.values()[i];
    if (init_weight_to_optim_[key]) {
      continue;
    } else {
      init_weight_to_optim_[key] = true;
    }
    ps_->InitWeightKeyToOptims(key, val);
  }
}

void ParameterServer::ServerHandler::HandleInitInputsShape(DataPtr data, size_t size, VectorPtr res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  const Key &key = input.keys()[0];
  if (init_optim_info_[key]) {
    return;
  } else {
    init_optim_info_[key] = true;
  }
  Keys keys = {input.keys().begin(), input.keys().end()};
  Values values = {input.values().begin(), input.values().end()};
  Lengths lens = {input.len().begin(), input.len().end()};
  ps_->InitOptimInputsShape(keys, values, lens);
}

void ParameterServer::ServerHandler::HandleInitEmbeddings(DataPtr data, size_t size, VectorPtr res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  EmbeddingTableMeta embedding_table_meta;
  embedding_table_meta.ParseFromArray(data.get(), size);
  const Key &key = embedding_table_meta.key();
  MS_LOG(INFO) << "Initializing embedding table for key:" << key;
  std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> shapes =
    std::make_shared<std::vector<std::shared_ptr<std::vector<size_t>>>>();
  MS_EXCEPTION_IF_NULL(shapes);
  std::shared_ptr<std::vector<size_t>> input_shape = std::make_shared<std::vector<size_t>>(
    embedding_table_meta.input_shape().begin(), embedding_table_meta.input_shape().end());
  MS_EXCEPTION_IF_NULL(input_shape);
  std::shared_ptr<std::vector<size_t>> indices_shape = std::make_shared<std::vector<size_t>>(
    embedding_table_meta.indices_shape().begin(), embedding_table_meta.indices_shape().end());
  MS_EXCEPTION_IF_NULL(indices_shape);
  std::shared_ptr<std::vector<size_t>> output_shape = std::make_shared<std::vector<size_t>>(
    embedding_table_meta.output_shape().begin(), embedding_table_meta.output_shape().end());
  MS_EXCEPTION_IF_NULL(output_shape);
  shapes->push_back(input_shape);
  shapes->push_back(indices_shape);
  shapes->push_back(output_shape);

  const ParamInitInfoMessage &info = embedding_table_meta.info();
  ParamInitInfo param_init_info;
  if (ps::PsDataPrefetch::GetInstance().cache_enable()) {
    param_init_info.param_type_ = static_cast<ParamType>(info.param_type());
    if (param_init_info.param_type_ == kWeight) {
      param_init_info.global_seed_ = info.global_seed();
      param_init_info.op_seed_ = info.op_seed();
    } else if (param_init_info.param_type_ == kAccumulation) {
      param_init_info.init_val_ = info.init_val();
    }
  }
  ps_->InitEmbeddingTable(key, shapes, param_init_info);
}

void ParameterServer::ServerHandler::HandleCheckReadyForPush(DataPtr data, size_t size, VectorPtr res) {
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  const Key &key = input.keys()[0];
  bool ready = ps_->ReadyForPush(key);
  MS_LOG(INFO) << "The ready is:" << ready;
  KVMessage res_data;
  res_data.add_keys(key);
  res_data.add_values(ready);
  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleCheckReadyForPull(DataPtr data, size_t size, VectorPtr res) {
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  const Key &key = input.keys()[0];
  bool ready = ps_->ReadyForPull(key);
  KVMessage res_data;
  res_data.add_keys(key);
  res_data.add_values(ready);
  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleEmbeddingLookup(DataPtr data, size_t size, VectorPtr res) {
  MS_EXCEPTION_IF_NULL(res);
  EmbeddingTableLookup input;
  input.ParseFromArray(data.get(), size);
  const Key &key = input.key();

  KVMessage res_data;
  std::vector<Key> keys = {input.keys().begin(), input.keys().end()};
  *res_data.mutable_keys() = {input.keys().begin(), input.keys().end()};

  ps_->DoEmbeddingLookup(key, keys, &res_data);

  res->resize(res_data.ByteSizeLong());
  size_t dest_size = res_data.ByteSizeLong();
  size_t src_size = res_data.ByteSizeLong();
  int ret = memcpy_s(res->data(), dest_size, res_data.SerializeAsString().data(), src_size);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
  }
}

void ParameterServer::ServerHandler::HandleUpdateEmbeddings(DataPtr data, size_t size, VectorPtr res) {
  std::unique_lock<std::mutex> lock(ps_->mutex());
  MS_EXCEPTION_IF_NULL(res);
  KVMessage input;
  input.ParseFromArray(data.get(), size);
  const Key &key = input.keys()[0];
  const LookupIds &lookup_ids = {input.keys().begin() + 1, input.keys().end()};
  const Values &update_vals = {input.values().begin(), input.values().end()};
  ps_->UpdateEmbeddings(key, lookup_ids, update_vals);
}

void ParameterServer::ServerHandler::HandleFinalize(DataPtr data, size_t size, VectorPtr res) {
  MS_EXCEPTION_IF_NULL(res);
  ps_->Finalize();
}
}  // namespace ps
}  // namespace mindspore
