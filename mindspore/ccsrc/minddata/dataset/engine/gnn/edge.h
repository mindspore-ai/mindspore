/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

class Edge {
 public:
  // Constructor
  // @param EdgeIdType id - edge id
  // @param EdgeType type - edge type
  // @param WeightType weight - edge weight
  // @param NodeIdType src_id - source node id
  // @param NodeIdType dst_id - destination node id
  Edge(EdgeIdType id, EdgeType type, WeightType weight, NodeIdType src_id, NodeIdType dst_id)
      : id_(id), type_(type), weight_(weight), src_id_(src_id), dst_id_(dst_id) {}

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
  // @param NodeIdType *src_id - Source node id returned
  // @param NodeIdType *dst_id - Destination node id returned
  Status GetNode(NodeIdType *src_id, NodeIdType *dst_id) {
    RETURN_UNEXPECTED_IF_NULL(src_id);
    RETURN_UNEXPECTED_IF_NULL(dst_id);
    *src_id = src_id_;
    *dst_id = dst_id_;
    return Status::OK();
  }

  // Set node to edge
  // @param NodeIdType src_id - Source node id
  // @param NodeIdType dst_id - Destination node id
  Status SetNode(NodeIdType src_id, NodeIdType dst_id) {
    src_id_ = src_id;
    dst_id_ = dst_id;
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
  NodeIdType src_id_;
  NodeIdType dst_id_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_EDGE_H_
