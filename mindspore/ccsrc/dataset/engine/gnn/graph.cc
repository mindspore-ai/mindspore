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
#include "dataset/engine/gnn/graph.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

#include "dataset/core/tensor_shape.h"

namespace mindspore {
namespace dataset {
namespace gnn {

Graph::Graph(std::string dataset_file, int32_t num_workers) : dataset_file_(dataset_file), num_workers_(num_workers) {
  MS_LOG(INFO) << "num_workers:" << num_workers;
}

Status Graph::GetNodes(NodeType node_type, NodeIdType node_num, std::shared_ptr<Tensor> *out) {
  auto itr = node_type_map_.find(node_type);
  if (itr == node_type_map_.end()) {
    std::string err_msg = "Invalid node type:" + std::to_string(node_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    if (node_num == -1) {
      RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>({itr->second}, DataType(DataType::DE_INT32), out));
    } else {
    }
  }
  return Status::OK();
}

template <typename T>
Status Graph::CreateTensorByVector(const std::vector<std::vector<T>> &data, DataType type,
                                   std::shared_ptr<Tensor> *out) {
  if (!type.IsCompatible<T>()) {
    RETURN_STATUS_UNEXPECTED("Data type not compatible");
  }
  if (data.empty()) {
    RETURN_STATUS_UNEXPECTED("Input data is emply");
  }
  std::shared_ptr<Tensor> tensor;
  size_t m = data.size();
  size_t n = data[0].size();
  RETURN_IF_NOT_OK(Tensor::CreateTensor(
    &tensor, TensorImpl::kFlexible, TensorShape({static_cast<dsize_t>(m), static_cast<dsize_t>(n)}), type, nullptr));
  T *ptr = reinterpret_cast<T *>(tensor->GetMutableBuffer());
  for (auto id_m : data) {
    CHECK_FAIL_RETURN_UNEXPECTED(id_m.size() == n, "Each member of the vector has a different size");
    for (auto id_n : id_m) {
      *ptr = id_n;
      ptr++;
    }
  }
  tensor->Squeeze();
  *out = std::move(tensor);
  return Status::OK();
}

template <typename T>
Status Graph::ComplementVector(std::vector<std::vector<T>> *data, size_t max_size, T default_value) {
  if (!data || data->empty()) {
    RETURN_STATUS_UNEXPECTED("Input data is emply");
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

Status Graph::GetEdges(EdgeType edge_type, EdgeIdType edge_num, std::shared_ptr<Tensor> *out) { return Status::OK(); }

Status Graph::GetAllNeighbors(const std::vector<NodeIdType> &node_list, NodeType neighbor_type,
                              std::shared_ptr<Tensor> *out) {
  if (node_type_map_.find(neighbor_type) == node_type_map_.end()) {
    std::string err_msg = "Invalid neighbor type:" + std::to_string(neighbor_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::vector<std::vector<NodeIdType>> neighbors;
  size_t max_neighbor_num = 0;
  neighbors.resize(node_list.size());
  for (size_t i = 0; i < node_list.size(); ++i) {
    auto itr = node_id_map_.find(node_list[i]);
    if (itr != node_id_map_.end()) {
      RETURN_IF_NOT_OK(itr->second->GetNeighbors(neighbor_type, -1, &neighbors[i]));
      max_neighbor_num = max_neighbor_num > neighbors[i].size() ? max_neighbor_num : neighbors[i].size();
    } else {
      std::string err_msg = "Invalid node id:" + std::to_string(node_list[i]);
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  RETURN_IF_NOT_OK(ComplementVector<NodeIdType>(&neighbors, max_neighbor_num, kDefaultNodeId));
  RETURN_IF_NOT_OK(CreateTensorByVector<NodeIdType>(neighbors, DataType(DataType::DE_INT32), out));

  return Status::OK();
}

Status Graph::GetSampledNeighbor(const std::vector<NodeIdType> &node_list, const std::vector<NodeIdType> &neighbor_nums,
                                 const std::vector<NodeType> &neighbor_types, std::shared_ptr<Tensor> *out) {
  return Status::OK();
}

Status Graph::GetNegSampledNeighbor(const std::vector<NodeIdType> &node_list, NodeIdType samples_num,
                                    NodeType neg_neighbor_type, std::shared_ptr<Tensor> *out) {
  return Status::OK();
}

Status Graph::RandomWalk(const std::vector<NodeIdType> &node_list, const std::vector<NodeType> &meta_path, float p,
                         float q, NodeIdType default_node, std::shared_ptr<Tensor> *out) {
  return Status::OK();
}

Status Graph::GetNodeDefaultFeature(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) {
  auto itr = default_feature_map_.find(feature_type);
  if (itr == default_feature_map_.end()) {
    std::string err_msg = "Invalid feature type:" + std::to_string(feature_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  } else {
    *out_feature = itr->second;
  }
  return Status::OK();
}

Status Graph::GetNodeFeature(const std::shared_ptr<Tensor> &nodes, const std::vector<FeatureType> &feature_types,
                             TensorRow *out) {
  if (!nodes || nodes->Size() == 0) {
    RETURN_STATUS_UNEXPECTED("Inpude nodes is empty");
  }
  TensorRow tensors;
  for (auto f_type : feature_types) {
    std::shared_ptr<Feature> default_feature;
    // If no feature can be obtained, fill in the default value
    RETURN_IF_NOT_OK(GetNodeDefaultFeature(f_type, &default_feature));

    TensorShape shape(default_feature->Value()->shape());
    auto shape_vec = nodes->shape().AsVector();
    dsize_t size = std::accumulate(shape_vec.begin(), shape_vec.end(), 1, std::multiplies<dsize_t>());
    shape = shape.PrependDim(size);
    std::shared_ptr<Tensor> fea_tensor;
    RETURN_IF_NOT_OK(
      Tensor::CreateTensor(&fea_tensor, TensorImpl::kFlexible, shape, default_feature->Value()->type(), nullptr));

    dsize_t index = 0;
    for (auto node_itr = nodes->begin<NodeIdType>(); node_itr != nodes->end<NodeIdType>(); ++node_itr) {
      auto itr = node_id_map_.find(*node_itr);
      std::shared_ptr<Feature> feature;
      if (itr != node_id_map_.end()) {
        if (!itr->second->GetFeatures(f_type, &feature).IsOk()) {
          feature = default_feature;
        }
      } else {
        if (*node_itr == kDefaultNodeId) {
          feature = default_feature;
        } else {
          std::string err_msg = "Invalid node id:" + std::to_string(*node_itr);
          RETURN_STATUS_UNEXPECTED(err_msg);
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

Status Graph::GetEdgeFeature(const std::shared_ptr<Tensor> &edges, const std::vector<FeatureType> &feature_types,
                             TensorRow *out) {
  return Status::OK();
}

Status Graph::Init() {
  RETURN_IF_NOT_OK(LoadNodeAndEdge());
  return Status::OK();
}

Status Graph::GetMetaInfo(std::vector<NodeMetaInfo> *node_info, std::vector<EdgeMetaInfo> *edge_info) {
  node_info->reserve(node_type_map_.size());
  for (auto node : node_type_map_) {
    NodeMetaInfo n_info;
    n_info.type = node.first;
    n_info.num = node.second.size();
    auto itr = node_feature_map_.find(node.first);
    if (itr != node_feature_map_.end()) {
      for (auto f_type : itr->second) {
        n_info.feature_type.push_back(f_type);
      }
      std::sort(n_info.feature_type.begin(), n_info.feature_type.end());
    }
    node_info->push_back(n_info);
  }

  edge_info->reserve(edge_type_map_.size());
  for (auto edge : edge_type_map_) {
    EdgeMetaInfo e_info;
    e_info.type = edge.first;
    e_info.num = edge.second.size();
    auto itr = edge_feature_map_.find(edge.first);
    if (itr != edge_feature_map_.end()) {
      for (auto f_type : itr->second) {
        e_info.feature_type.push_back(f_type);
      }
    }
    edge_info->push_back(e_info);
  }
  return Status::OK();
}

Status Graph::LoadNodeAndEdge() {
  GraphLoader gl(dataset_file_, num_workers_);
  // ask graph_loader to load everything into memory
  RETURN_IF_NOT_OK(gl.InitAndLoad());
  // get all maps
  RETURN_IF_NOT_OK(gl.GetNodesAndEdges(&node_id_map_, &edge_id_map_, &node_type_map_, &edge_type_map_,
                                       &node_feature_map_, &edge_feature_map_, &default_feature_map_));
  return Status::OK();
}
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
