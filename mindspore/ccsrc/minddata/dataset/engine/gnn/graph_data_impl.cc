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
#include "minddata/dataset/engine/gnn/graph_data_impl.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <utility>

#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/gnn/graph_loader.h"
#include "minddata/dataset/util/random.h"
namespace mindspore {
namespace dataset {
namespace gnn {

GraphDataImpl::GraphDataImpl(std::string dataset_file, int32_t num_workers, bool server_mode)
    : dataset_file_(dataset_file),
      num_workers_(num_workers),
      rnd_(GetRandomDevice()),
      random_walk_(this),
      server_mode_(server_mode) {
  rnd_.seed(GetSeed());
  MS_LOG(INFO) << "num_workers:" << num_workers;
}

GraphDataImpl::~GraphDataImpl() {}

Status GraphDataImpl::GetAllNodes(NodeType node_type, std::shared_ptr<Tensor> *out) {
  auto itr = node_type_map_.find(node_type);
  if (itr == node_type_map_.end()) {
    std::string err_msg = "Invalid node type:" + std::to_string(node_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>({itr->second}, DataType(DataType::DE_INT32), out));
  }
  return Status::OK();
}

template <typename T>
Status GraphDataImpl::CreateTensorByVector(const std::vector<std::vector<T>> &data, DataType type,
                                           std::shared_ptr<Tensor> *out) {
  if (!type.IsCompatible<T>()) {
    RETURN_STATUS_UNEXPECTED("Data type not compatible");
  }
  if (data.empty()) {
    RETURN_STATUS_UNEXPECTED("Input data is empty");
  }
  std::shared_ptr<Tensor> tensor;
  size_t m = data.size();
  size_t n = data[0].size();
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({static_cast<dsize_t>(m), static_cast<dsize_t>(n)}), type, &tensor));
  auto ptr = tensor->begin<T>();
  for (const auto &id_m : data) {
    CHECK_FAIL_RETURN_UNEXPECTED(id_m.size() == n, "Each member of the vector has a different size");
    for (const auto &id_n : id_m) {
      *ptr = id_n;
      ptr++;
    }
  }
  tensor->Squeeze();
  *out = std::move(tensor);
  return Status::OK();
}

template <typename T>
Status GraphDataImpl::ComplementVector(std::vector<std::vector<T>> *data, size_t max_size, T default_value) {
  if (!data || data->empty()) {
    RETURN_STATUS_UNEXPECTED("Input data is empty");
  }
  for (std::vector<T> &vec : *data) {
    size_t size = vec.size();
    if (size > max_size) {
      RETURN_STATUS_UNEXPECTED("The max_size parameter is abnormal");
    } else {
      for (size_t i = 0; i < (max_size - size); ++i) {
        vec.push_back(default_value);
      }
    }
  }
  return Status::OK();
}

Status GraphDataImpl::GetAllEdges(EdgeType edge_type, std::shared_ptr<Tensor> *out) {
  auto itr = edge_type_map_.find(edge_type);
  if (itr == edge_type_map_.end()) {
    std::string err_msg = "Invalid edge type:" + std::to_string(edge_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    RETURN_IF_NOT_OK(CreateTensorByVector<EdgeIdType>({itr->second}, DataType(DataType::DE_INT32), out));
  }
  return Status::OK();
}

Status GraphDataImpl::GetNodesFromEdges(const std::vector<EdgeIdType> &edge_list, std::shared_ptr<Tensor> *out) {
  if (edge_list.empty()) {
    RETURN_STATUS_UNEXPECTED("Input edge_list is empty");
  }

  std::vector<std::vector<NodeIdType>> node_list;
  node_list.reserve(edge_list.size());
  for (const auto &edge_id : edge_list) {
    auto itr = edge_id_map_.find(edge_id);
    if (itr == edge_id_map_.end()) {
      std::string err_msg = "Invalid edge id:" + std::to_string(edge_id);
      RETURN_STATUS_UNEXPECTED(err_msg);
    } else {
      std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> nodes;
      RETURN_IF_NOT_OK(itr->second->GetNode(&nodes));
      node_list.push_back({nodes.first->id(), nodes.second->id()});
    }
  }
  RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>(node_list, DataType(DataType::DE_INT32), out));
  return Status::OK();
}

Status GraphDataImpl::GetAllNeighbors(const std::vector<NodeIdType> &node_list, NodeType neighbor_type,
                                      std::shared_ptr<Tensor> *out) {
  CHECK_FAIL_RETURN_UNEXPECTED(!node_list.empty(), "Input node_list is empty.");
  RETURN_IF_NOT_OK(CheckNeighborType(neighbor_type));

  std::vector<std::vector<NodeIdType>> neighbors;
  size_t max_neighbor_num = 0;
  neighbors.resize(node_list.size());
  for (size_t i = 0; i < node_list.size(); ++i) {
    std::shared_ptr<Node> node;
    RETURN_IF_NOT_OK(GetNodeByNodeId(node_list[i], &node));
    RETURN_IF_NOT_OK(node->GetAllNeighbors(neighbor_type, &neighbors[i]));
    max_neighbor_num = max_neighbor_num > neighbors[i].size() ? max_neighbor_num : neighbors[i].size();
  }

  RETURN_IF_NOT_OK(ComplementVector<NodeIdType>(&neighbors, max_neighbor_num, kDefaultNodeId));
  RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>(neighbors, DataType(DataType::DE_INT32), out));

  return Status::OK();
}

Status GraphDataImpl::CheckSamplesNum(NodeIdType samples_num) {
  NodeIdType all_nodes_number =
    std::accumulate(node_type_map_.begin(), node_type_map_.end(), 0,
                    [](NodeIdType t1, const auto &t2) -> NodeIdType { return t1 + t2.second.size(); });
  if ((samples_num < 1) || (samples_num > all_nodes_number)) {
    std::string err_msg = "Wrong samples number, should be between 1 and " + std::to_string(all_nodes_number) +
                          ", got " + std::to_string(samples_num);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status GraphDataImpl::CheckNeighborType(NodeType neighbor_type) {
  if (node_type_map_.find(neighbor_type) == node_type_map_.end()) {
    std::string err_msg = "Invalid neighbor type:" + std::to_string(neighbor_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

Status GraphDataImpl::GetSampledNeighbors(const std::vector<NodeIdType> &node_list,
                                          const std::vector<NodeIdType> &neighbor_nums,
                                          const std::vector<NodeType> &neighbor_types, SamplingStrategy strategy,
                                          std::shared_ptr<Tensor> *out) {
  CHECK_FAIL_RETURN_UNEXPECTED(!node_list.empty(), "Input node_list is empty.");
  CHECK_FAIL_RETURN_UNEXPECTED(neighbor_nums.size() == neighbor_types.size(),
                               "The sizes of neighbor_nums and neighbor_types are inconsistent.");
  for (const auto &num : neighbor_nums) {
    RETURN_IF_NOT_OK(CheckSamplesNum(num));
  }
  for (const auto &type : neighbor_types) {
    RETURN_IF_NOT_OK(CheckNeighborType(type));
  }
  std::vector<std::vector<NodeIdType>> neighbors_vec(node_list.size());
  for (size_t node_idx = 0; node_idx < node_list.size(); ++node_idx) {
    std::shared_ptr<Node> input_node;
    RETURN_IF_NOT_OK(GetNodeByNodeId(node_list[node_idx], &input_node));
    neighbors_vec[node_idx].emplace_back(node_list[node_idx]);
    std::vector<NodeIdType> input_list = {node_list[node_idx]};
    for (size_t i = 0; i < neighbor_nums.size(); ++i) {
      std::vector<NodeIdType> neighbors;
      neighbors.reserve(input_list.size() * neighbor_nums[i]);
      for (const auto &node_id : input_list) {
        if (node_id == kDefaultNodeId) {
          for (int32_t j = 0; j < neighbor_nums[i]; ++j) {
            neighbors.emplace_back(kDefaultNodeId);
          }
        } else {
          std::shared_ptr<Node> node;
          RETURN_IF_NOT_OK(GetNodeByNodeId(node_id, &node));
          std::vector<NodeIdType> out;
          RETURN_IF_NOT_OK(node->GetSampledNeighbors(neighbor_types[i], neighbor_nums[i], strategy, &out));
          neighbors.insert(neighbors.end(), out.begin(), out.end());
        }
      }
      neighbors_vec[node_idx].insert(neighbors_vec[node_idx].end(), neighbors.begin(), neighbors.end());
      input_list = std::move(neighbors);
    }
  }
  RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>(neighbors_vec, DataType(DataType::DE_INT32), out));
  return Status::OK();
}

Status GraphDataImpl::NegativeSample(const std::vector<NodeIdType> &data, const std::vector<NodeIdType> shuffled_ids,
                                     size_t *start_index, const std::unordered_set<NodeIdType> &exclude_data,
                                     int32_t samples_num, std::vector<NodeIdType> *out_samples) {
  CHECK_FAIL_RETURN_UNEXPECTED(!data.empty(), "Input data is empty.");
  size_t index = *start_index;
  for (size_t i = index; i < shuffled_ids.size(); ++i) {
    ++index;
    if (exclude_data.find(data[shuffled_ids[i]]) != exclude_data.end()) {
      continue;
    }
    out_samples->emplace_back(data[shuffled_ids[i]]);
    if (out_samples->size() >= samples_num) {
      break;
    }
  }
  *start_index = index;
  return Status::OK();
}

Status GraphDataImpl::GetNegSampledNeighbors(const std::vector<NodeIdType> &node_list, NodeIdType samples_num,
                                             NodeType neg_neighbor_type, std::shared_ptr<Tensor> *out) {
  CHECK_FAIL_RETURN_UNEXPECTED(!node_list.empty(), "Input node_list is empty.");
  RETURN_IF_NOT_OK(CheckSamplesNum(samples_num));
  RETURN_IF_NOT_OK(CheckNeighborType(neg_neighbor_type));

  const std::vector<NodeIdType> &all_nodes = node_type_map_[neg_neighbor_type];
  std::vector<NodeIdType> shuffled_id(all_nodes.size());
  std::iota(shuffled_id.begin(), shuffled_id.end(), 0);
  std::shuffle(shuffled_id.begin(), shuffled_id.end(), rnd_);
  size_t start_index = 0;
  bool need_shuffle = false;

  std::vector<std::vector<NodeIdType>> neg_neighbors_vec;
  neg_neighbors_vec.resize(node_list.size());
  for (size_t node_idx = 0; node_idx < node_list.size(); ++node_idx) {
    std::shared_ptr<Node> node;
    RETURN_IF_NOT_OK(GetNodeByNodeId(node_list[node_idx], &node));
    std::vector<NodeIdType> neighbors;
    RETURN_IF_NOT_OK(node->GetAllNeighbors(neg_neighbor_type, &neighbors));
    std::unordered_set<NodeIdType> exclude_nodes;
    std::transform(neighbors.begin(), neighbors.end(),
                   std::insert_iterator<std::unordered_set<NodeIdType>>(exclude_nodes, exclude_nodes.begin()),
                   [](const NodeIdType node) { return node; });
    neg_neighbors_vec[node_idx].emplace_back(node->id());
    if (all_nodes.size() > exclude_nodes.size()) {
      while (neg_neighbors_vec[node_idx].size() < samples_num + 1) {
        RETURN_IF_NOT_OK(NegativeSample(all_nodes, shuffled_id, &start_index, exclude_nodes, samples_num + 1,
                                        &neg_neighbors_vec[node_idx]));
        if (start_index >= shuffled_id.size()) {
          start_index = start_index % shuffled_id.size();
          need_shuffle = true;
        }
      }
    } else {
      MS_LOG(DEBUG) << "There are no negative neighbors. node_id:" << node->id()
                    << " neg_neighbor_type:" << neg_neighbor_type;
      // If there are no negative neighbors, they are filled with kDefaultNodeId
      for (int32_t i = 0; i < samples_num; ++i) {
        neg_neighbors_vec[node_idx].emplace_back(kDefaultNodeId);
      }
    }
    if (need_shuffle) {
      std::shuffle(shuffled_id.begin(), shuffled_id.end(), rnd_);
      start_index = 0;
      need_shuffle = false;
    }
  }
  RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>(neg_neighbors_vec, DataType(DataType::DE_INT32), out));
  return Status::OK();
}

Status GraphDataImpl::RandomWalk(const std::vector<NodeIdType> &node_list, const std::vector<NodeType> &meta_path,
                                 float step_home_param, float step_away_param, NodeIdType default_node,
                                 std::shared_ptr<Tensor> *out) {
  RETURN_IF_NOT_OK(random_walk_.Build(node_list, meta_path, step_home_param, step_away_param, default_node));
  std::vector<std::vector<NodeIdType>> walks;
  RETURN_IF_NOT_OK(random_walk_.SimulateWalk(&walks));
  RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>({walks}, DataType(DataType::DE_INT32), out));
  return Status::OK();
}

Status GraphDataImpl::GetNodeDefaultFeature(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) {
  auto itr = default_node_feature_map_.find(feature_type);
  if (itr == default_node_feature_map_.end()) {
    std::string err_msg = "Invalid feature type:" + std::to_string(feature_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    *out_feature = itr->second;
  }
  return Status::OK();
}

Status GraphDataImpl::GetEdgeDefaultFeature(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) {
  auto itr = default_edge_feature_map_.find(feature_type);
  if (itr == default_edge_feature_map_.end()) {
    std::string err_msg = "Invalid feature type:" + std::to_string(feature_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    *out_feature = itr->second;
  }
  return Status::OK();
}

Status GraphDataImpl::GetNodeFeature(const std::shared_ptr<Tensor> &nodes,
                                     const std::vector<FeatureType> &feature_types, TensorRow *out) {
  if (!nodes || nodes->Size() == 0) {
    RETURN_STATUS_UNEXPECTED("Input nodes is empty");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!feature_types.empty(), "Input feature_types is empty");
  TensorRow tensors;
  for (const auto &f_type : feature_types) {
    std::shared_ptr<Feature> default_feature;
    // If no feature can be obtained, fill in the default value
    RETURN_IF_NOT_OK(GetNodeDefaultFeature(f_type, &default_feature));

    TensorShape shape(default_feature->Value()->shape());
    auto shape_vec = nodes->shape().AsVector();
    dsize_t size = std::accumulate(shape_vec.begin(), shape_vec.end(), 1, std::multiplies<dsize_t>());
    shape = shape.PrependDim(size);
    std::shared_ptr<Tensor> fea_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, default_feature->Value()->type(), &fea_tensor));

    dsize_t index = 0;
    for (auto node_itr = nodes->begin<NodeIdType>(); node_itr != nodes->end<NodeIdType>(); ++node_itr) {
      std::shared_ptr<Feature> feature;
      if (*node_itr == kDefaultNodeId) {
        feature = default_feature;
      } else {
        std::shared_ptr<Node> node;
        RETURN_IF_NOT_OK(GetNodeByNodeId(*node_itr, &node));
        if (!node->GetFeatures(f_type, &feature).IsOk()) {
          feature = default_feature;
        }
      }
      RETURN_IF_NOT_OK(fea_tensor->InsertTensor({index}, feature->Value()));
      index++;
    }

    TensorShape reshape(nodes->shape());
    for (auto s : default_feature->Value()->shape().AsVector()) {
      reshape = reshape.AppendDim(s);
    }
    RETURN_IF_NOT_OK(fea_tensor->Reshape(reshape));
    fea_tensor->Squeeze();
    tensors.push_back(fea_tensor);
  }
  *out = std::move(tensors);
  return Status::OK();
}

Status GraphDataImpl::GetNodeFeatureSharedMemory(const std::shared_ptr<Tensor> &nodes, FeatureType type,
                                                 std::shared_ptr<Tensor> *out) {
  if (!nodes || nodes->Size() == 0) {
    RETURN_STATUS_UNEXPECTED("Input nodes is empty");
  }
  TensorShape shape = nodes->shape().AppendDim(2);
  std::shared_ptr<Tensor> fea_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, DataType(DataType::DE_INT64), &fea_tensor));

  auto out_fea_itr = fea_tensor->begin<int64_t>();
  for (auto node_itr = nodes->begin<NodeIdType>(); node_itr != nodes->end<NodeIdType>(); ++node_itr) {
    if (*node_itr == kDefaultNodeId) {
      *out_fea_itr = -1;
      ++out_fea_itr;
      *out_fea_itr = -1;
      ++out_fea_itr;
    } else {
      std::shared_ptr<Node> node;
      RETURN_IF_NOT_OK(GetNodeByNodeId(*node_itr, &node));
      std::shared_ptr<Feature> feature;
      if (!node->GetFeatures(type, &feature).IsOk()) {
        *out_fea_itr = -1;
        ++out_fea_itr;
        *out_fea_itr = -1;
        ++out_fea_itr;
      } else {
        for (auto fea_itr = feature->Value()->begin<int64_t>(); fea_itr != feature->Value()->end<int64_t>();
             ++fea_itr) {
          *out_fea_itr = *fea_itr;
          ++out_fea_itr;
        }
      }
    }
  }

  fea_tensor->Squeeze();

  *out = std::move(fea_tensor);
  return Status::OK();
}

Status GraphDataImpl::GetEdgeFeature(const std::shared_ptr<Tensor> &edges,
                                     const std::vector<FeatureType> &feature_types, TensorRow *out) {
  if (!edges || edges->Size() == 0) {
    RETURN_STATUS_UNEXPECTED("Input edges is empty");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!feature_types.empty(), "Input feature_types is empty");
  TensorRow tensors;
  for (const auto &f_type : feature_types) {
    std::shared_ptr<Feature> default_feature;
    // If no feature can be obtained, fill in the default value
    RETURN_IF_NOT_OK(GetEdgeDefaultFeature(f_type, &default_feature));

    TensorShape shape(default_feature->Value()->shape());
    auto shape_vec = edges->shape().AsVector();
    dsize_t size = std::accumulate(shape_vec.begin(), shape_vec.end(), 1, std::multiplies<dsize_t>());
    shape = shape.PrependDim(size);
    std::shared_ptr<Tensor> fea_tensor;
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, default_feature->Value()->type(), &fea_tensor));

    dsize_t index = 0;
    for (auto edge_itr = edges->begin<EdgeIdType>(); edge_itr != edges->end<EdgeIdType>(); ++edge_itr) {
      std::shared_ptr<Edge> edge;
      RETURN_IF_NOT_OK(GetEdgeByEdgeId(*edge_itr, &edge));
      std::shared_ptr<Feature> feature;
      if (!edge->GetFeatures(f_type, &feature).IsOk()) {
        feature = default_feature;
      }
      RETURN_IF_NOT_OK(fea_tensor->InsertTensor({index}, feature->Value()));
      index++;
    }

    TensorShape reshape(edges->shape());
    for (auto s : default_feature->Value()->shape().AsVector()) {
      reshape = reshape.AppendDim(s);
    }
    RETURN_IF_NOT_OK(fea_tensor->Reshape(reshape));
    fea_tensor->Squeeze();
    tensors.push_back(fea_tensor);
  }
  *out = std::move(tensors);
  return Status::OK();
}

Status GraphDataImpl::GetEdgeFeatureSharedMemory(const std::shared_ptr<Tensor> &edges, FeatureType type,
                                                 std::shared_ptr<Tensor> *out) {
  if (!edges || edges->Size() == 0) {
    RETURN_STATUS_UNEXPECTED("Input edges is empty");
  }
  TensorShape shape = edges->shape().AppendDim(2);
  std::shared_ptr<Tensor> fea_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(shape, DataType(DataType::DE_INT64), &fea_tensor));

  auto out_fea_itr = fea_tensor->begin<int64_t>();
  for (auto edge_itr = edges->begin<EdgeIdType>(); edge_itr != edges->end<EdgeIdType>(); ++edge_itr) {
    std::shared_ptr<Edge> edge;
    RETURN_IF_NOT_OK(GetEdgeByEdgeId(*edge_itr, &edge));
    std::shared_ptr<Feature> feature;
    if (!edge->GetFeatures(type, &feature).IsOk()) {
      *out_fea_itr = -1;
      ++out_fea_itr;
      *out_fea_itr = -1;
      ++out_fea_itr;
    } else {
      for (auto fea_itr = feature->Value()->begin<int64_t>(); fea_itr != feature->Value()->end<int64_t>(); ++fea_itr) {
        *out_fea_itr = *fea_itr;
        ++out_fea_itr;
      }
    }
  }

  fea_tensor->Squeeze();

  *out = std::move(fea_tensor);
  return Status::OK();
}

Status GraphDataImpl::Init() {
  RETURN_IF_NOT_OK(LoadNodeAndEdge());
  return Status::OK();
}

Status GraphDataImpl::GetMetaInfo(MetaInfo *meta_info) {
  meta_info->node_type.resize(node_type_map_.size());
  std::transform(node_type_map_.begin(), node_type_map_.end(), meta_info->node_type.begin(),
                 [](auto itr) { return itr.first; });
  std::sort(meta_info->node_type.begin(), meta_info->node_type.end());

  meta_info->edge_type.resize(edge_type_map_.size());
  std::transform(edge_type_map_.begin(), edge_type_map_.end(), meta_info->edge_type.begin(),
                 [](auto itr) { return itr.first; });
  std::sort(meta_info->edge_type.begin(), meta_info->edge_type.end());

  for (const auto &node : node_type_map_) {
    meta_info->node_num[node.first] = node.second.size();
  }

  for (const auto &edge : edge_type_map_) {
    meta_info->edge_num[edge.first] = edge.second.size();
  }

  for (const auto &node_feature : node_feature_map_) {
    for (auto type : node_feature.second) {
      meta_info->node_feature_type.emplace_back(type);
    }
  }
  std::sort(meta_info->node_feature_type.begin(), meta_info->node_feature_type.end());
  auto unique_node = std::unique(meta_info->node_feature_type.begin(), meta_info->node_feature_type.end());
  meta_info->node_feature_type.erase(unique_node, meta_info->node_feature_type.end());

  for (const auto &edge_feature : edge_feature_map_) {
    for (const auto &type : edge_feature.second) {
      meta_info->edge_feature_type.emplace_back(type);
    }
  }
  std::sort(meta_info->edge_feature_type.begin(), meta_info->edge_feature_type.end());
  auto unique_edge = std::unique(meta_info->edge_feature_type.begin(), meta_info->edge_feature_type.end());
  meta_info->edge_feature_type.erase(unique_edge, meta_info->edge_feature_type.end());
  return Status::OK();
}

#ifdef ENABLE_PYTHON
Status GraphDataImpl::GraphInfo(py::dict *out) {
  MetaInfo meta_info;
  RETURN_IF_NOT_OK(GetMetaInfo(&meta_info));
  (*out)["node_type"] = py::cast(meta_info.node_type);
  (*out)["edge_type"] = py::cast(meta_info.edge_type);
  (*out)["node_num"] = py::cast(meta_info.node_num);
  (*out)["edge_num"] = py::cast(meta_info.edge_num);
  (*out)["node_feature_type"] = py::cast(meta_info.node_feature_type);
  (*out)["edge_feature_type"] = py::cast(meta_info.edge_feature_type);
  return Status::OK();
}
#endif

Status GraphDataImpl::LoadNodeAndEdge() {
  GraphLoader gl(this, dataset_file_, num_workers_, server_mode_);
  // ask graph_loader to load everything into memory
  RETURN_IF_NOT_OK(gl.InitAndLoad());
  // get all maps
  RETURN_IF_NOT_OK(gl.GetNodesAndEdges());
  return Status::OK();
}

Status GraphDataImpl::GetNodeByNodeId(NodeIdType id, std::shared_ptr<Node> *node) {
  auto itr = node_id_map_.find(id);
  if (itr == node_id_map_.end()) {
    std::string err_msg = "Invalid node id:" + std::to_string(id);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    *node = itr->second;
  }
  return Status::OK();
}

Status GraphDataImpl::GetEdgeByEdgeId(EdgeIdType id, std::shared_ptr<Edge> *edge) {
  auto itr = edge_id_map_.find(id);
  if (itr == edge_id_map_.end()) {
    std::string err_msg = "Invalid edge id:" + std::to_string(id);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    *edge = itr->second;
  }
  return Status::OK();
}

GraphDataImpl::RandomWalkBase::RandomWalkBase(GraphDataImpl *graph)
    : graph_(graph), step_home_param_(1.0), step_away_param_(1.0), default_node_(-1), num_walks_(1), num_workers_(1) {}

Status GraphDataImpl::RandomWalkBase::Build(const std::vector<NodeIdType> &node_list,
                                            const std::vector<NodeType> &meta_path, float step_home_param,
                                            float step_away_param, const NodeIdType default_node, int32_t num_walks,
                                            int32_t num_workers) {
  CHECK_FAIL_RETURN_UNEXPECTED(!node_list.empty(), "Input node_list is empty.");
  node_list_ = node_list;
  if (meta_path.empty() || meta_path.size() > kMaxNumWalks) {
    std::string err_msg = "Failed, meta path required between 1 and " + std::to_string(kMaxNumWalks) +
                          ". The size of input path is " + std::to_string(meta_path.size());
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  for (const auto &type : meta_path) {
    RETURN_IF_NOT_OK(graph_->CheckNeighborType(type));
  }
  meta_path_ = meta_path;
  if (step_home_param < kGnnEpsilon || step_away_param < kGnnEpsilon) {
    std::string err_msg = "Failed, step_home_param and step_away_param required greater than " +
                          std::to_string(kGnnEpsilon) + ". step_home_param: " + std::to_string(step_home_param) +
                          ", step_away_param: " + std::to_string(step_away_param);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (default_node < -1) {
    std::string err_msg = "Failed, default_node required to be greater or equal to -1.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (num_walks <= 0) {
    std::string err_msg = "Failed, num_walks parameter required to be greater than 0";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (num_workers <= 0) {
    std::string err_msg = "Failed, num_workers parameter required to be greater than 0";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  step_home_param_ = step_home_param;
  step_away_param_ = step_away_param;
  default_node_ = default_node;
  num_walks_ = num_walks;
  num_workers_ = num_workers;
  return Status::OK();
}

Status GraphDataImpl::RandomWalkBase::Node2vecWalk(const NodeIdType &start_node, std::vector<NodeIdType> *walk_path) {
  // Simulate a random walk starting from start node.
  auto walk = std::vector<NodeIdType>(1, start_node);  // walk is an vector
  // walk simulate
  while (walk.size() - 1 < meta_path_.size()) {
    // current nodE
    auto cur_node_id = walk.back();
    std::shared_ptr<Node> cur_node;
    RETURN_IF_NOT_OK(graph_->GetNodeByNodeId(cur_node_id, &cur_node));

    // current neighbors
    std::vector<NodeIdType> cur_neighbors;
    RETURN_IF_NOT_OK(cur_node->GetAllNeighbors(meta_path_[walk.size() - 1], &cur_neighbors, true));
    std::sort(cur_neighbors.begin(), cur_neighbors.end());

    // break if no neighbors
    if (cur_neighbors.empty()) {
      break;
    }

    // walk by the fist node, then by the previous 2 nodes
    std::shared_ptr<StochasticIndex> stochastic_index;
    if (walk.size() == 1) {
      RETURN_IF_NOT_OK(GetNodeProbability(cur_node_id, meta_path_[0], &stochastic_index));
    } else {
      NodeIdType prev_node_id = walk[walk.size() - 2];
      RETURN_IF_NOT_OK(GetEdgeProbability(prev_node_id, cur_node_id, walk.size() - 2, &stochastic_index));
    }
    NodeIdType next_node_id = cur_neighbors[WalkToNextNode(*stochastic_index)];
    walk.push_back(next_node_id);
  }

  while (walk.size() - 1 < meta_path_.size()) {
    walk.push_back(default_node_);
  }

  *walk_path = std::move(walk);
  return Status::OK();
}

Status GraphDataImpl::RandomWalkBase::SimulateWalk(std::vector<std::vector<NodeIdType>> *walks) {
  for (int32_t i = 0; i < num_walks_; ++i) {
    for (const auto &node : node_list_) {
      std::vector<NodeIdType> walk;
      RETURN_IF_NOT_OK(Node2vecWalk(node, &walk));
      walks->push_back(walk);
    }
  }
  return Status::OK();
}

Status GraphDataImpl::RandomWalkBase::GetNodeProbability(const NodeIdType &node_id, const NodeType &node_type,
                                                         std::shared_ptr<StochasticIndex> *node_probability) {
  // Generate alias nodes
  std::shared_ptr<Node> node;
  graph_->GetNodeByNodeId(node_id, &node);
  std::vector<NodeIdType> neighbors;
  RETURN_IF_NOT_OK(node->GetAllNeighbors(node_type, &neighbors, true));
  std::sort(neighbors.begin(), neighbors.end());
  auto non_normalized_probability = std::vector<float>(neighbors.size(), 1.0);
  *node_probability =
    std::make_shared<StochasticIndex>(GenerateProbability(Normalize<float>(non_normalized_probability)));
  return Status::OK();
}

Status GraphDataImpl::RandomWalkBase::GetEdgeProbability(const NodeIdType &src, const NodeIdType &dst,
                                                         uint32_t meta_path_index,
                                                         std::shared_ptr<StochasticIndex> *edge_probability) {
  // Get the alias edge setup lists for a given edge.
  std::shared_ptr<Node> src_node;
  graph_->GetNodeByNodeId(src, &src_node);
  std::vector<NodeIdType> src_neighbors;
  RETURN_IF_NOT_OK(src_node->GetAllNeighbors(meta_path_[meta_path_index], &src_neighbors, true));

  std::shared_ptr<Node> dst_node;
  graph_->GetNodeByNodeId(dst, &dst_node);
  std::vector<NodeIdType> dst_neighbors;
  RETURN_IF_NOT_OK(dst_node->GetAllNeighbors(meta_path_[meta_path_index + 1], &dst_neighbors, true));

  std::sort(dst_neighbors.begin(), dst_neighbors.end());
  std::vector<float> non_normalized_probability;
  for (const auto &dst_nbr : dst_neighbors) {
    if (dst_nbr == src) {
      non_normalized_probability.push_back(1.0 / step_home_param_);  // replace 1.0 with G[dst][dst_nbr]['weight']
      continue;
    }
    auto it = std::find(src_neighbors.begin(), src_neighbors.end(), dst_nbr);
    if (it != src_neighbors.end()) {
      // stay close, this node connect both src and dst
      non_normalized_probability.push_back(1.0);  // replace 1.0 with G[dst][dst_nbr]['weight']
    } else {
      // step far away
      non_normalized_probability.push_back(1.0 / step_away_param_);  // replace 1.0 with G[dst][dst_nbr]['weight']
    }
  }

  *edge_probability =
    std::make_shared<StochasticIndex>(GenerateProbability(Normalize<float>(non_normalized_probability)));
  return Status::OK();
}

StochasticIndex GraphDataImpl::RandomWalkBase::GenerateProbability(const std::vector<float> &probability) {
  uint32_t K = probability.size();
  std::vector<int32_t> switch_to_large_index(K, 0);
  std::vector<float> weight(K, .0);
  std::vector<int32_t> smaller;
  std::vector<int32_t> larger;
  auto random_device = GetRandomDevice();
  std::uniform_real_distribution<> distribution(-kGnnEpsilon, kGnnEpsilon);
  float accumulate_threshold = 0.0;
  for (uint32_t i = 0; i < K; i++) {
    float threshold_one = distribution(random_device);
    accumulate_threshold += threshold_one;
    weight[i] = i < K - 1 ? probability[i] * K + threshold_one : probability[i] * K - accumulate_threshold;
    weight[i] < 1.0 ? smaller.push_back(i) : larger.push_back(i);
  }

  while ((!smaller.empty()) && (!larger.empty())) {
    uint32_t small = smaller.back();
    smaller.pop_back();
    uint32_t large = larger.back();
    larger.pop_back();
    switch_to_large_index[small] = large;
    weight[large] = weight[large] + weight[small] - 1.0;
    weight[large] < 1.0 ? smaller.push_back(large) : larger.push_back(large);
  }
  return StochasticIndex(switch_to_large_index, weight);
}

uint32_t GraphDataImpl::RandomWalkBase::WalkToNextNode(const StochasticIndex &stochastic_index) {
  auto switch_to_large_index = stochastic_index.first;
  auto weight = stochastic_index.second;
  const uint32_t size_of_index = switch_to_large_index.size();

  auto random_device = GetRandomDevice();
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  // Generate random integer between [0, K)
  uint32_t random_idx = std::floor(distribution(random_device) * size_of_index);

  if (distribution(random_device) < weight[random_idx]) {
    return random_idx;
  }
  return switch_to_large_index[random_idx];
}

template <typename T>
std::vector<float> GraphDataImpl::RandomWalkBase::Normalize(const std::vector<T> &non_normalized_probability) {
  float sum_probability =
    1.0 * std::accumulate(non_normalized_probability.begin(), non_normalized_probability.end(), 0);
  if (sum_probability < kGnnEpsilon) {
    sum_probability = 1.0;
  }
  std::vector<float> normalized_probability;
  std::transform(non_normalized_probability.begin(), non_normalized_probability.end(),
                 std::back_inserter(normalized_probability), [&](T value) -> float { return value / sum_probability; });
  return normalized_probability;
}
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
