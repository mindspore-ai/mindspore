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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_EDGE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_EDGE_H_

#include <memory>
#include <unordered_map>
#include <utility>

#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/gnn/feature.h"
#include "minddata/dataset/engine/gnn/node.h"

namespace mindspore {
namespace dataset {
namespace gnn {
using EdgeType = int8_t;
using EdgeIdType = int32_t;

class Edge {
 public:
  // Constructor
  // @param EdgeIdType id - edge id
  // @param EdgeType type - edge type
  // @param WeightType weight - edge weight
  // @param std::shared_ptr<Node> src_node - source node
  // @param std::shared_ptr<Node> dst_node - destination node
  Edge(EdgeIdType id, EdgeType type, WeightType weight, std::shared_ptr<Node> src_node, std::shared_ptr<Node> dst_node)
      : id_(id), type_(type), weight_(weight), src_node_(src_node), dst_node_(dst_node) {}

  virtual ~Edge() = default;

  // @return NodeIdType - Returned edge id
  EdgeIdType id() const { return id_; }

  // @return NodeIdType - Returned edge type
  EdgeType type() const { return type_; }

  // @return WeightType - Returned edge weight
  WeightType weight() const { return weight_; }

  // Get the feature of a edge
  // @param FeatureType feature_type - type of feature
  // @param std::shared_ptr<Feature> *out_feature - Returned feature
  // @return Status The status code returned
  virtual Status GetFeatures(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) = 0;

  // Get nodes on the edge
  // @param std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> *out_node - Source and destination nodes returned
  Status GetNode(std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> *out_node) {
    *out_node = std::make_pair(src_node_, dst_node_);
    return Status::OK();
  }

  // Set node to edge
  // @param const std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> &in_node -
  Status SetNode(const std::pair<std::shared_ptr<Node>, std::shared_ptr<Node>> &in_node) {
    src_node_ = in_node.first;
    dst_node_ = in_node.second;
    return Status::OK();
  }

  // Update feature of edge
  // @param std::shared_ptr<Feature> feature -
  // @return Status The status code returned
  virtual Status UpdateFeature(const std::shared_ptr<Feature> &feature) = 0;

 protected:
  EdgeIdType id_;
  EdgeType type_;
  WeightType weight_;
  std::shared_ptr<Node> src_node_;
  std::shared_ptr<Node> dst_node_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_EDGE_H_
