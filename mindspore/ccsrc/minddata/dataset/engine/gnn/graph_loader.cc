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
#include "minddata/dataset/engine/gnn/graph_loader.h"

#include <future>
#include <tuple>
#include <utility>

#include "minddata/dataset/engine/gnn/graph_data_impl.h"
#include "minddata/dataset/engine/gnn/local_edge.h"
#include "minddata/dataset/engine/gnn/local_node.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/mindrecord/include/shard_error.h"

using ShardTuple = std::vector<std::tuple<std::vector<uint8_t>, mindspore::mindrecord::json>>;
namespace mindspore {
namespace dataset {
namespace gnn {

using mindrecord::MSRStatus;

GraphLoader::GraphLoader(GraphDataImpl *graph_impl, std::string mr_filepath, int32_t num_workers, bool server_mode)
    : graph_impl_(graph_impl),
      mr_path_(mr_filepath),
      num_workers_(num_workers),
      row_id_(0),
      shard_reader_(nullptr),
      graph_feature_parser_(nullptr),
      required_key_(
        {"first_id", "second_id", "third_id", "attribute", "type", "node_feature_index", "edge_feature_index"}),
      optional_key_({{"weight", false}}) {}

Status GraphLoader::GetNodesAndEdges() {
  NodeIdMap *n_id_map = &graph_impl_->node_id_map_;
  EdgeIdMap *e_id_map = &graph_impl_->edge_id_map_;
  for (std::deque<std::shared_ptr<Node>> &dq : n_deques_) {
    while (dq.empty() == false) {
      std::shared_ptr<Node> node_ptr = dq.front();
      n_id_map->insert({node_ptr->id(), node_ptr});
      graph_impl_->node_type_map_[node_ptr->type()].push_back(node_ptr->id());
      dq.pop_front();
    }
  }

  for (std::deque<std::shared_ptr<Edge>> &dq : e_deques_) {
    while (dq.empty() == false) {
      std::shared_ptr<Edge> edge_ptr = dq.front();
      std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> p;
      RETURN_IF_NOT_OK(edge_ptr->GetNode(&p));
      auto src_itr = n_id_map->find(p.first->id()), dst_itr = n_id_map->find(p.second->id());
      CHECK_FAIL_RETURN_UNEXPECTED(src_itr != n_id_map->end(), "invalid src_id:" + std::to_string(src_itr->first));
      CHECK_FAIL_RETURN_UNEXPECTED(dst_itr != n_id_map->end(), "invalid src_id:" + std::to_string(dst_itr->first));
      RETURN_IF_NOT_OK(edge_ptr->SetNode({src_itr->second, dst_itr->second}));
      RETURN_IF_NOT_OK(src_itr->second->AddNeighbor(dst_itr->second, edge_ptr->weight()));
      e_id_map->insert({edge_ptr->id(), edge_ptr});  // add edge to edge_id_map_
      graph_impl_->edge_type_map_[edge_ptr->type()].push_back(edge_ptr->id());
      dq.pop_front();
    }
  }

  for (auto &itr : graph_impl_->node_type_map_) itr.second.shrink_to_fit();
  for (auto &itr : graph_impl_->edge_type_map_) itr.second.shrink_to_fit();

  MergeFeatureMaps();
  return Status::OK();
}

Status GraphLoader::InitAndLoad() {
  CHECK_FAIL_RETURN_UNEXPECTED(num_workers_ > 0, "num_reader can't be < 1\n");
  CHECK_FAIL_RETURN_UNEXPECTED(row_id_ == 0, "InitAndLoad Can only be called once!\n");
  n_deques_.resize(num_workers_);
  e_deques_.resize(num_workers_);
  n_feature_maps_.resize(num_workers_);
  e_feature_maps_.resize(num_workers_);
  default_node_feature_maps_.resize(num_workers_);
  default_edge_feature_maps_.resize(num_workers_);
  TaskGroup vg;

  shard_reader_ = std::make_unique<ShardReader>();
  CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_->Open({mr_path_}, true, num_workers_) == MSRStatus::SUCCESS,
                               "Fail to open" + mr_path_);
  CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_->GetShardHeader()->GetSchemaCount() > 0, "No schema found!");
  CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_->Launch(true) == MSRStatus::SUCCESS, "fail to launch mr");

  graph_impl_->data_schema_ = (shard_reader_->GetShardHeader()->GetSchemas()[0]->GetSchema());
  mindrecord::json schema = graph_impl_->data_schema_["schema"];
  for (const std::string &key : required_key_) {
    if (schema.find(key) == schema.end()) {
      RETURN_STATUS_UNEXPECTED(key + ":doesn't exist in schema:" + schema.dump());
    }
  }

  for (auto op_key : optional_key_) {
    if (schema.find(op_key.first) != schema.end()) {
      optional_key_[op_key.first] = true;
    }
  }

  if (graph_impl_->server_mode_) {
#if !defined(_WIN32) && !defined(_WIN64)
    int64_t total_blob_size = 0;
    CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_->GetTotalBlobSize(&total_blob_size) == MSRStatus::SUCCESS,
                                 "failed to get total blob size");
    graph_impl_->graph_shared_memory_ = std::make_unique<GraphSharedMemory>(total_blob_size, mr_path_);
    RETURN_IF_NOT_OK(graph_impl_->graph_shared_memory_->CreateSharedMemory());
#endif
  }

  graph_feature_parser_ = std::make_unique<GraphFeatureParser>(*shard_reader_->GetShardColumn());

  // launching worker threads
  for (int wkr_id = 0; wkr_id < num_workers_; ++wkr_id) {
    RETURN_IF_NOT_OK(vg.CreateAsyncTask("GraphLoader", std::bind(&GraphLoader::WorkerEntry, this, wkr_id)));
  }
  // wait for threads to finish and check its return code
  vg.join_all(Task::WaitFlag::kBlocking);
  RETURN_IF_NOT_OK(vg.GetTaskErrorIfAny());
  return Status::OK();
}

Status GraphLoader::LoadNode(const std::vector<uint8_t> &col_blob, const mindrecord::json &col_jsn,
                             std::shared_ptr<Node> *node, NodeFeatureMap *feature_map,
                             DefaultNodeFeatureMap *default_feature) {
  NodeIdType node_id = col_jsn["first_id"];
  NodeType node_type = static_cast<NodeType>(col_jsn["type"]);
  WeightType weight = 1;
  if (optional_key_["weight"]) {
    weight = col_jsn["weight"];
  }
  (*node) = std::make_shared<LocalNode>(node_id, node_type, weight);
  std::vector<int32_t> indices;
  RETURN_IF_NOT_OK(graph_feature_parser_->LoadFeatureIndex("node_feature_index", col_blob, &indices));
  if (graph_impl_->server_mode_) {
#if !defined(_WIN32) && !defined(_WIN64)
    for (int32_t ind : indices) {
      std::shared_ptr<Tensor> tensor_sm;
      RETURN_IF_NOT_OK(graph_feature_parser_->LoadFeatureToSharedMemory(
        "node_feature_" + std::to_string(ind), col_blob, graph_impl_->graph_shared_memory_.get(), &tensor_sm));
      RETURN_IF_NOT_OK((*node)->UpdateFeature(std::make_shared<Feature>(ind, tensor_sm, true)));
      (*feature_map)[node_type].insert(ind);
      if ((*default_feature)[ind] == nullptr) {
        std::shared_ptr<Tensor> tensor;
        RETURN_IF_NOT_OK(
          graph_feature_parser_->LoadFeatureTensor("node_feature_" + std::to_string(ind), col_blob, &tensor));
        std::shared_ptr<Tensor> zero_tensor;
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(tensor->shape(), tensor->type(), &zero_tensor));
        RETURN_IF_NOT_OK(zero_tensor->Zero());
        (*default_feature)[ind] = std::make_shared<Feature>(ind, zero_tensor);
      }
    }
#endif
  } else {
    for (int32_t ind : indices) {
      std::shared_ptr<Tensor> tensor;
      RETURN_IF_NOT_OK(
        graph_feature_parser_->LoadFeatureTensor("node_feature_" + std::to_string(ind), col_blob, &tensor));
      RETURN_IF_NOT_OK((*node)->UpdateFeature(std::make_shared<Feature>(ind, tensor)));
      (*feature_map)[node_type].insert(ind);
      if ((*default_feature)[ind] == nullptr) {
        std::shared_ptr<Tensor> zero_tensor;
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(tensor->shape(), tensor->type(), &zero_tensor));
        RETURN_IF_NOT_OK(zero_tensor->Zero());
        (*default_feature)[ind] = std::make_shared<Feature>(ind, zero_tensor);
      }
    }
  }
  return Status::OK();
}

Status GraphLoader::LoadEdge(const std::vector<uint8_t> &col_blob, const mindrecord::json &col_jsn,
                             std::shared_ptr<Edge> *edge, EdgeFeatureMap *feature_map,
                             DefaultEdgeFeatureMap *default_feature) {
  EdgeIdType edge_id = col_jsn["first_id"];
  EdgeType edge_type = static_cast<EdgeType>(col_jsn["type"]);
  NodeIdType src_id = col_jsn["second_id"], dst_id = col_jsn["third_id"];
  WeightType edge_weight = 1;
  if (optional_key_["weight"]) {
    edge_weight = col_jsn["weight"];
  }
  std::shared_ptr<Node> src = std::make_shared<LocalNode>(src_id, -1, 1);
  std::shared_ptr<Node> dst = std::make_shared<LocalNode>(dst_id, -1, 1);
  (*edge) = std::make_shared<LocalEdge>(edge_id, edge_type, edge_weight, src, dst);
  std::vector<int32_t> indices;
  RETURN_IF_NOT_OK(graph_feature_parser_->LoadFeatureIndex("edge_feature_index", col_blob, &indices));
  if (graph_impl_->server_mode_) {
#if !defined(_WIN32) && !defined(_WIN64)
    for (int32_t ind : indices) {
      std::shared_ptr<Tensor> tensor_sm;
      RETURN_IF_NOT_OK(graph_feature_parser_->LoadFeatureToSharedMemory(
        "edge_feature_" + std::to_string(ind), col_blob, graph_impl_->graph_shared_memory_.get(), &tensor_sm));
      RETURN_IF_NOT_OK((*edge)->UpdateFeature(std::make_shared<Feature>(ind, tensor_sm, true)));
      (*feature_map)[edge_type].insert(ind);
      if ((*default_feature)[ind] == nullptr) {
        std::shared_ptr<Tensor> tensor;
        RETURN_IF_NOT_OK(
          graph_feature_parser_->LoadFeatureTensor("edge_feature_" + std::to_string(ind), col_blob, &tensor));
        std::shared_ptr<Tensor> zero_tensor;
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(tensor->shape(), tensor->type(), &zero_tensor));
        RETURN_IF_NOT_OK(zero_tensor->Zero());
        (*default_feature)[ind] = std::make_shared<Feature>(ind, zero_tensor);
      }
    }
#endif
  } else {
    for (int32_t ind : indices) {
      std::shared_ptr<Tensor> tensor;
      RETURN_IF_NOT_OK(
        graph_feature_parser_->LoadFeatureTensor("edge_feature_" + std::to_string(ind), col_blob, &tensor));
      RETURN_IF_NOT_OK((*edge)->UpdateFeature(std::make_shared<Feature>(ind, tensor)));
      (*feature_map)[edge_type].insert(ind);
      if ((*default_feature)[ind] == nullptr) {
        std::shared_ptr<Tensor> zero_tensor;
        RETURN_IF_NOT_OK(Tensor::CreateEmpty(tensor->shape(), tensor->type(), &zero_tensor));
        RETURN_IF_NOT_OK(zero_tensor->Zero());
        (*default_feature)[ind] = std::make_shared<Feature>(ind, zero_tensor);
      }
    }
  }

  return Status::OK();
}

Status GraphLoader::WorkerEntry(int32_t worker_id) {
  // Handshake
  TaskManager::FindMe()->Post();
  auto ret = shard_reader_->GetNextById(row_id_++, worker_id);
  ShardTuple rows = ret.second;
  while (rows.empty() == false) {
    RETURN_IF_INTERRUPTED();
    for (const auto &tupled_row : rows) {
      std::vector<uint8_t> col_blob = std::get<0>(tupled_row);
      mindrecord::json col_jsn = std::get<1>(tupled_row);
      std::string attr = col_jsn["attribute"];
      if (attr == "n") {
        std::shared_ptr<Node> node_ptr;
        RETURN_IF_NOT_OK(LoadNode(col_blob, col_jsn, &node_ptr, &(n_feature_maps_[worker_id]),
                                  &default_node_feature_maps_[worker_id]));
        n_deques_[worker_id].emplace_back(node_ptr);
      } else if (attr == "e") {
        std::shared_ptr<Edge> edge_ptr;
        RETURN_IF_NOT_OK(LoadEdge(col_blob, col_jsn, &edge_ptr, &(e_feature_maps_[worker_id]),
                                  &default_edge_feature_maps_[worker_id]));
        e_deques_[worker_id].emplace_back(edge_ptr);
      } else {
        MS_LOG(WARNING) << "attribute:" << attr << " is neither edge nor node.";
      }
    }
    auto rc = shard_reader_->GetNextById(row_id_++, worker_id);
    rows = rc.second;
  }
  return Status::OK();
}

void GraphLoader::MergeFeatureMaps() {
  for (int wkr_id = 0; wkr_id < num_workers_; wkr_id++) {
    for (auto &m : n_feature_maps_[wkr_id]) {
      for (auto &n : m.second) graph_impl_->node_feature_map_[m.first].insert(n);
    }
    for (auto &m : e_feature_maps_[wkr_id]) {
      for (auto &n : m.second) graph_impl_->edge_feature_map_[m.first].insert(n);
    }
    for (auto &m : default_node_feature_maps_[wkr_id]) {
      graph_impl_->default_node_feature_map_[m.first] = m.second;
    }
    for (auto &m : default_edge_feature_maps_[wkr_id]) {
      graph_impl_->default_edge_feature_map_[m.first] = m.second;
    }
  }
  n_feature_maps_.clear();
  e_feature_maps_.clear();
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
