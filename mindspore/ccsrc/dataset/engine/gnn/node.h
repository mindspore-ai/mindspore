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
#ifndef DATASET_ENGINE_GNN_NODE_H_
#define DATASET_ENGINE_GNN_NODE_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "dataset/util/status.h"
#include "dataset/engine/gnn/feature.h"

namespace mindspore {
namespace dataset {
namespace gnn {
using NodeType = int8_t;
using NodeIdType = int32_t;

constexpr NodeIdType kDefaultNodeId = -1;

class Node {
 public:
  // Constructor
  // @param NodeIdType id - node id
  // @param NodeType type - node type
  Node(NodeIdType id, NodeType type) : id_(id), type_(type) {}

  virtual ~Node() = default;

  // @return NodeIdType - Returned node id
  NodeIdType id() const { return id_; }

  // @return NodeIdType - Returned node type
  NodeType type() const { return type_; }

  // Get the feature of a node
  // @param FeatureType feature_type - type of feature
  // @param std::shared_ptr<Feature> *out_feature - Returned feature
  // @return Status - The error code return
  virtual Status GetFeatures(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) = 0;

  // Get the all neighbors of a node
  // @param NodeType neighbor_type - type of neighbor
  // @param std::vector<NodeIdType> *out_neighbors - Returned neighbors id
  // @return Status - The error code return
  virtual Status GetAllNeighbors(NodeType neighbor_type, std::vector<NodeIdType> *out_neighbors) = 0;

  // Get the sampled neighbors of a node
  // @param NodeType neighbor_type - type of neighbor
  // @param int32_t samples_num - Number of neighbors to be acquired
  // @param std::vector<NodeIdType> *out_neighbors - Returned neighbors id
  // @return Status - The error code return
  virtual Status GetSampledNeighbors(NodeType neighbor_type, int32_t samples_num,
                                     std::vector<NodeIdType> *out_neighbors) = 0;

  // Add neighbor of node
  // @param std::shared_ptr<Node> node -
  // @return Status - The error code return
  virtual Status AddNeighbor(const std::shared_ptr<Node> &node) = 0;

  // Update feature of node
  // @param std::shared_ptr<Feature> feature -
  // @return Status - The error code return
  virtual Status UpdateFeature(const std::shared_ptr<Feature> &feature) = 0;

 protected:
  NodeIdType id_;
  NodeType type_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_GNN_NODE_H_
