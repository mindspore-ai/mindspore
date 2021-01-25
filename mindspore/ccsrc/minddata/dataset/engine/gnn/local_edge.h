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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_LOCAL_EDGE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_LOCAL_EDGE_H_

#include <memory>
#include <unordered_map>
#include <utility>

#include "minddata/dataset/engine/gnn/edge.h"
#include "minddata/dataset/engine/gnn/feature.h"
#include "minddata/dataset/engine/gnn/node.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
namespace gnn {

class LocalEdge : public Edge {
 public:
  // Constructor
  // @param EdgeIdType id - edge id
  // @param EdgeType type - edge type
  // @param WeightType weight - edge weight
  // @param std::shared_ptr<Node> src_node - source node
  // @param std::shared_ptr<Node> dst_node - destination node
  LocalEdge(EdgeIdType id, EdgeType type, WeightType weight, std::shared_ptr<Node> src_node,
            std::shared_ptr<Node> dst_node);

  ~LocalEdge() = default;

  // Get the feature of a edge
  // @param FeatureType feature_type - type of feature
  // @param std::shared_ptr<Feature> *out_feature - Returned feature
  // @return Status The status code returned
  Status GetFeatures(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) override;

  // Update feature of edge
  // @param std::shared_ptr<Feature> feature -
  // @return Status The status code returned
  Status UpdateFeature(const std::shared_ptr<Feature> &feature) override;

 private:
  std::unordered_map<FeatureType, std::shared_ptr<Feature>> features_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_LOCAL_EDGE_H_
