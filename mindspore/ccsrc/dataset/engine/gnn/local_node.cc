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
#include "dataset/engine/gnn/local_node.h"

#include <algorithm>
#include <string>
#include <utility>

#include "dataset/engine/gnn/edge.h"
#include "dataset/util/random.h"

namespace mindspore {
namespace dataset {
namespace gnn {

LocalNode::LocalNode(NodeIdType id, NodeType type) : Node(id, type), rnd_(GetRandomDevice()) { rnd_.seed(GetSeed()); }

Status LocalNode::GetFeatures(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) {
  auto itr = features_.find(feature_type);
  if (itr != features_.end()) {
    *out_feature = itr->second;
    return Status::OK();
  } else {
    std::string err_msg = "Invalid feature type:" + std::to_string(feature_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
}

Status LocalNode::GetAllNeighbors(NodeType neighbor_type, std::vector<NodeIdType> *out_neighbors) {
  std::vector<NodeIdType> neighbors;
  auto itr = neighbor_nodes_.find(neighbor_type);
  if (itr != neighbor_nodes_.end()) {
    neighbors.resize(itr->second.size() + 1);
    neighbors[0] = id_;
    std::transform(itr->second.begin(), itr->second.end(), neighbors.begin() + 1,
                   [](const std::shared_ptr<Node> node) { return node->id(); });
  } else {
    MS_LOG(DEBUG) << "No neighbors. node_id:" << id_ << " neighbor_type:" << neighbor_type;
    neighbors.emplace_back(id_);
  }
  *out_neighbors = std::move(neighbors);
  return Status::OK();
}

Status LocalNode::GetSampledNeighbors(const std::vector<std::shared_ptr<Node>> &neighbors, int32_t samples_num,
                                      std::vector<NodeIdType> *out) {
  std::vector<NodeIdType> shuffled_id(neighbors.size());
  std::iota(shuffled_id.begin(), shuffled_id.end(), 0);
  std::shuffle(shuffled_id.begin(), shuffled_id.end(), rnd_);
  int32_t num = std::min(samples_num, static_cast<int32_t>(neighbors.size()));
  for (int32_t i = 0; i < num; ++i) {
    out->emplace_back(neighbors[shuffled_id[i]]->id());
  }
  return Status::OK();
}

Status LocalNode::GetSampledNeighbors(NodeType neighbor_type, int32_t samples_num,
                                      std::vector<NodeIdType> *out_neighbors) {
  std::vector<NodeIdType> neighbors;
  neighbors.reserve(samples_num);
  auto itr = neighbor_nodes_.find(neighbor_type);
  if (itr != neighbor_nodes_.end()) {
    while (neighbors.size() < samples_num) {
      RETURN_IF_NOT_OK(GetSampledNeighbors(itr->second, samples_num - neighbors.size(), &neighbors));
    }
  } else {
    MS_LOG(DEBUG) << "There are no neighbors. node_id:" << id_ << " neighbor_type:" << neighbor_type;
    // If there are no neighbors, they are filled with kDefaultNodeId
    for (int32_t i = 0; i < samples_num; ++i) {
      neighbors.emplace_back(kDefaultNodeId);
    }
  }
  *out_neighbors = std::move(neighbors);
  return Status::OK();
}

Status LocalNode::AddNeighbor(const std::shared_ptr<Node> &node) {
  auto itr = neighbor_nodes_.find(node->type());
  if (itr != neighbor_nodes_.end()) {
    itr->second.push_back(node);
  } else {
    neighbor_nodes_[node->type()] = {node};
  }
  return Status::OK();
}

Status LocalNode::UpdateFeature(const std::shared_ptr<Feature> &feature) {
  auto itr = features_.find(feature->type());
  if (itr != features_.end()) {
    RETURN_STATUS_UNEXPECTED("Feature already exists");
  } else {
    features_[feature->type()] = feature;
    return Status::OK();
  }
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
