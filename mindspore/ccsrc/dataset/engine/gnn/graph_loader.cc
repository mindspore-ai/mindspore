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

#include <future>
#include <tuple>
#include <utility>

#include "dataset/engine/gnn/graph_loader.h"
#include "mindspore/ccsrc/mindrecord/include/shard_error.h"
#include "dataset/engine/gnn/local_edge.h"
#include "dataset/engine/gnn/local_node.h"

using ShardTuple = std::vector<std::tuple<std::vector<uint8_t>, mindspore::mindrecord::json>>;

namespace mindspore {
namespace dataset {
namespace gnn {

using mindrecord::MSRStatus;

GraphLoader::GraphLoader(std::string mr_filepath, int32_t num_workers)
    : mr_path_(mr_filepath),
      num_workers_(num_workers),
      row_id_(0),
      keys_({"first_id", "second_id", "third_id", "attribute", "type", "node_feature_index", "edge_feature_index"}) {}

Status GraphLoader::GetNodesAndEdges(NodeIdMap *n_id_map, EdgeIdMap *e_id_map, NodeTypeMap *n_type_map,
                                     EdgeTypeMap *e_type_map, NodeFeatureMap *n_feature_map,
                                     EdgeFeatureMap *e_feature_map, DefaultFeatureMap *default_feature_map) {
  for (std::deque<std::shared_ptr<Node>> &dq : n_deques_) {
    while (dq.empty() == false) {
      std::shared_ptr<Node> node_ptr = dq.front();
      n_id_map->insert({node_ptr->id(), node_ptr});
      (*n_type_map)[node_ptr->type()].push_back(node_ptr->id());
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
      RETURN_IF_NOT_OK(src_itr->second->AddNeighbor(dst_itr->second));
      e_id_map->insert({edge_ptr->id(), edge_ptr});  // add edge to edge_id_map_
      (*e_type_map)[edge_ptr->type()].push_back(edge_ptr->id());
      dq.pop_front();
    }
  }

  for (auto &itr : *n_type_map) itr.second.shrink_to_fit();
  for (auto &itr : *e_type_map) itr.second.shrink_to_fit();

  MergeFeatureMaps(n_feature_map, e_feature_map, default_feature_map);
  return Status::OK();
}

Status GraphLoader::InitAndLoad() {
  CHECK_FAIL_RETURN_UNEXPECTED(num_workers_ > 0, "num_reader can't be < 1\n");
  CHECK_FAIL_RETURN_UNEXPECTED(row_id_ == 0, "InitAndLoad Can only be called once!\n");
  n_deques_.resize(num_workers_);
  e_deques_.resize(num_workers_);
  n_feature_maps_.resize(num_workers_);
  e_feature_maps_.resize(num_workers_);
  default_feature_maps_.resize(num_workers_);
  std::vector<std::future<Status>> r_codes(num_workers_);

  shard_reader_ = std::make_unique<ShardReader>();
  CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_->Open({mr_path_}, true, num_workers_) == MSRStatus::SUCCESS,
                               "Fail to open" + mr_path_);
  CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_->GetShardHeader()->GetSchemaCount() > 0, "No schema found!");
  CHECK_FAIL_RETURN_UNEXPECTED(shard_reader_->Launch(true) == MSRStatus::SUCCESS, "fail to launch mr");

  mindrecord::json schema = (shard_reader_->GetShardHeader()->GetSchemas()[0]->GetSchema())["schema"];
  for (const std::string &key : keys_) {
    if (schema.find(key) == schema.end()) {
      RETURN_STATUS_UNEXPECTED(key + ":doesn't exist in schema:" + schema.dump());
    }
  }

  // launching worker threads
  for (int wkr_id = 0; wkr_id < num_workers_; ++wkr_id) {
    r_codes[wkr_id] = std::async(std::launch::async, &GraphLoader::WorkerEntry, this, wkr_id);
  }
  // wait for threads to finish and check its return code
  for (int wkr_id = 0; wkr_id < num_workers_; ++wkr_id) {
    RETURN_IF_NOT_OK(r_codes[wkr_id].get());
  }
  return Status::OK();
}

Status GraphLoader::LoadNode(const std::vector<uint8_t> &col_blob, const mindrecord::json &col_jsn,
                             std::shared_ptr<Node> *node, NodeFeatureMap *feature_map,
                             DefaultFeatureMap *default_feature) {
  NodeIdType node_id = col_jsn["first_id"];
  NodeType node_type = static_cast<NodeType>(col_jsn["type"]);
  (*node) = std::make_shared<LocalNode>(node_id, node_type);
  std::vector<int32_t> indices;
  RETURN_IF_NOT_OK(LoadFeatureIndex("node_feature_index", col_blob, col_jsn, &indices));

  for (int32_t ind : indices) {
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(LoadFeatureTensor("node_feature_" + std::to_string(ind), col_blob, col_jsn, &tensor));
    RETURN_IF_NOT_OK((*node)->UpdateFeature(std::make_shared<Feature>(ind, tensor)));
    (*feature_map)[node_type].insert(ind);
    if ((*default_feature)[ind] == nullptr) {
      std::shared_ptr<Tensor> zero_tensor;
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&zero_tensor, TensorImpl::kFlexible, tensor->shape(), tensor->type()));
      RETURN_IF_NOT_OK(zero_tensor->Zero());
      (*default_feature)[ind] = std::make_shared<Feature>(ind, zero_tensor);
    }
  }
  return Status::OK();
}

Status GraphLoader::LoadEdge(const std::vector<uint8_t> &col_blob, const mindrecord::json &col_jsn,
                             std::shared_ptr<Edge> *edge, EdgeFeatureMap *feature_map,
                             DefaultFeatureMap *default_feature) {
  EdgeIdType edge_id = col_jsn["first_id"];
  EdgeType edge_type = static_cast<EdgeType>(col_jsn["type"]);
  NodeIdType src_id = col_jsn["second_id"], dst_id = col_jsn["third_id"];
  std::shared_ptr<Node> src = std::make_shared<LocalNode>(src_id, -1);
  std::shared_ptr<Node> dst = std::make_shared<LocalNode>(dst_id, -1);
  (*edge) = std::make_shared<LocalEdge>(edge_id, edge_type, src, dst);
  std::vector<int32_t> indices;
  RETURN_IF_NOT_OK(LoadFeatureIndex("edge_feature_index", col_blob, col_jsn, &indices));
  for (int32_t ind : indices) {
    std::shared_ptr<Tensor> tensor;
    RETURN_IF_NOT_OK(LoadFeatureTensor("edge_feature_" + std::to_string(ind), col_blob, col_jsn, &tensor));
    RETURN_IF_NOT_OK((*edge)->UpdateFeature(std::make_shared<Feature>(ind, tensor)));
    (*feature_map)[edge_type].insert(ind);
    if ((*default_feature)[ind] == nullptr) {
      std::shared_ptr<Tensor> zero_tensor;
      RETURN_IF_NOT_OK(Tensor::CreateTensor(&zero_tensor, TensorImpl::kFlexible, tensor->shape(), tensor->type()));
      RETURN_IF_NOT_OK(zero_tensor->Zero());
      (*default_feature)[ind] = std::make_shared<Feature>(ind, zero_tensor);
    }
  }
  return Status::OK();
}

Status GraphLoader::LoadFeatureTensor(const std::string &key, const std::vector<uint8_t> &col_blob,
                                      const mindrecord::json &col_jsn, std::shared_ptr<Tensor> *tensor) {
  const unsigned char *data = nullptr;
  std::unique_ptr<unsigned char[]> data_ptr;
  uint64_t n_bytes = 0, col_type_size = 1;
  mindrecord::ColumnDataType col_type = mindrecord::ColumnNoDataType;
  std::vector<int64_t> column_shape;
  MSRStatus rs = shard_reader_->GetShardColumn()->GetColumnValueByName(
    key, col_blob, col_jsn, &data, &data_ptr, &n_bytes, &col_type, &col_type_size, &column_shape);
  CHECK_FAIL_RETURN_UNEXPECTED(rs == mindrecord::SUCCESS, "fail to load column" + key);
  if (data == nullptr) data = reinterpret_cast<const unsigned char *>(&data_ptr[0]);
  RETURN_IF_NOT_OK(Tensor::CreateTensor(tensor, TensorImpl::kFlexible,
                                        std::move(TensorShape({static_cast<dsize_t>(n_bytes / col_type_size)})),
                                        std::move(DataType(mindrecord::ColumnDataTypeNameNormalized[col_type])), data));
  return Status::OK();
}

Status GraphLoader::LoadFeatureIndex(const std::string &key, const std::vector<uint8_t> &col_blob,
                                     const mindrecord::json &col_jsn, std::vector<int32_t> *indices) {
  const unsigned char *data = nullptr;
  std::unique_ptr<unsigned char[]> data_ptr;
  uint64_t n_bytes = 0, col_type_size = 1;
  mindrecord::ColumnDataType col_type = mindrecord::ColumnNoDataType;
  std::vector<int64_t> column_shape;
  MSRStatus rs = shard_reader_->GetShardColumn()->GetColumnValueByName(
    key, col_blob, col_jsn, &data, &data_ptr, &n_bytes, &col_type, &col_type_size, &column_shape);
  CHECK_FAIL_RETURN_UNEXPECTED(rs == mindrecord::SUCCESS, "fail to load column:" + key);

  if (data == nullptr) data = reinterpret_cast<const unsigned char *>(&data_ptr[0]);

  for (int i = 0; i < n_bytes; i += col_type_size) {
    int32_t feature_ind = -1;
    if (col_type == mindrecord::ColumnInt32) {
      feature_ind = *(reinterpret_cast<const int32_t *>(data + i));
    } else if (col_type == mindrecord::ColumnInt64) {
      feature_ind = *(reinterpret_cast<const int64_t *>(data + i));
    } else {
      RETURN_STATUS_UNEXPECTED("Feature Index needs to be int32/int64 type!");
    }
    if (feature_ind >= 0) indices->push_back(feature_ind);
  }
  return Status::OK();
}

Status GraphLoader::WorkerEntry(int32_t worker_id) {
  ShardTuple rows = shard_reader_->GetNextById(row_id_++, worker_id);
  while (rows.empty() == false) {
    for (const auto &tupled_row : rows) {
      std::vector<uint8_t> col_blob = std::get<0>(tupled_row);
      mindrecord::json col_jsn = std::get<1>(tupled_row);
      std::string attr = col_jsn["attribute"];
      if (attr == "n") {
        std::shared_ptr<Node> node_ptr;
        RETURN_IF_NOT_OK(
          LoadNode(col_blob, col_jsn, &node_ptr, &(n_feature_maps_[worker_id]), &default_feature_maps_[worker_id]));
        n_deques_[worker_id].emplace_back(node_ptr);
      } else if (attr == "e") {
        std::shared_ptr<Edge> edge_ptr;
        RETURN_IF_NOT_OK(
          LoadEdge(col_blob, col_jsn, &edge_ptr, &(e_feature_maps_[worker_id]), &default_feature_maps_[worker_id]));
        e_deques_[worker_id].emplace_back(edge_ptr);
      } else {
        MS_LOG(WARNING) << "attribute:" << attr << " is neither edge nor node.";
      }
    }
    rows = shard_reader_->GetNextById(row_id_++, worker_id);
  }
  return Status::OK();
}

void GraphLoader::MergeFeatureMaps(NodeFeatureMap *n_feature_map, EdgeFeatureMap *e_feature_map,
                                   DefaultFeatureMap *default_feature_map) {
  for (int wkr_id = 0; wkr_id < num_workers_; wkr_id++) {
    for (auto &m : n_feature_maps_[wkr_id]) {
      for (auto &n : m.second) (*n_feature_map)[m.first].insert(n);
    }
    for (auto &m : e_feature_maps_[wkr_id]) {
      for (auto &n : m.second) (*e_feature_map)[m.first].insert(n);
    }
    for (auto &m : default_feature_maps_[wkr_id]) {
      (*default_feature_map)[m.first] = m.second;
    }
  }
  n_feature_maps_.clear();
  e_feature_maps_.clear();
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
