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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/engine/gnn/feature.h"
#include "minddata/dataset/engine/gnn/node.h"
#include "minddata/dataset/engine/gnn/edge.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
namespace gnn {

struct MetaInfo {
  std::vector<NodeType> node_type;
  std::vector<EdgeType> edge_type;
  std::map<NodeType, NodeIdType> node_num;
  std::map<EdgeType, EdgeIdType> edge_num;
  std::vector<FeatureType> node_feature_type;
  std::vector<FeatureType> edge_feature_type;
};

class GraphData {
 public:
  // Get all nodes from the graph.
  // @param NodeType node_type - type of node
  // @param std::shared_ptr<Tensor> *out - Returned nodes id
  // @return Status The status code returned
  virtual Status GetAllNodes(NodeType node_type, std::shared_ptr<Tensor> *out) = 0;

  // Get all edges from the graph.
  // @param NodeType edge_type - type of edge
  // @param std::shared_ptr<Tensor> *out - Returned edge ids
  // @return Status The status code returned
  virtual Status GetAllEdges(EdgeType edge_type, std::shared_ptr<Tensor> *out) = 0;

  // Get the node id from the edge.
  // @param std::vector<EdgeIdType> edge_list - List of edges
  // @param std::shared_ptr<Tensor> *out - Returned node ids
  // @return Status The status code returned
  virtual Status GetNodesFromEdges(const std::vector<EdgeIdType> &edge_list, std::shared_ptr<Tensor> *out) = 0;

  // All neighbors of the acquisition node.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param NodeType neighbor_type - The type of neighbor. If the type does not exist, an error will be reported
  // @param std::shared_ptr<Tensor> *out - Returned neighbor's id. Because the number of neighbors at different nodes is
  // different, the returned tensor is output according to the maximum number of neighbors. If the number of neighbors
  // is not enough, fill in tensor as -1.
  // @return Status The status code returned
  virtual Status GetAllNeighbors(const std::vector<NodeIdType> &node_list, NodeType neighbor_type,
                                 std::shared_ptr<Tensor> *out) = 0;

  // Get sampled neighbors.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param std::vector<NodeIdType> neighbor_nums - Number of neighbors sampled per hop
  // @param std::vector<NodeType> neighbor_types - Neighbor type sampled per hop
  // @param std::SamplingStrategy strategy - Sampling strategy
  // @param std::shared_ptr<Tensor> *out - Returned neighbor's id.
  // @return Status The status code returned
  virtual Status GetSampledNeighbors(const std::vector<NodeIdType> &node_list,
                                     const std::vector<NodeIdType> &neighbor_nums,
                                     const std::vector<NodeType> &neighbor_types, SamplingStrategy strategy,
                                     std::shared_ptr<Tensor> *out) = 0;

  // Get negative sampled neighbors.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param NodeIdType samples_num - Number of neighbors sampled
  // @param NodeType neg_neighbor_type - The type of negative neighbor.
  // @param std::shared_ptr<Tensor> *out - Returned negative neighbor's id.
  // @return Status The status code returned
  virtual Status GetNegSampledNeighbors(const std::vector<NodeIdType> &node_list, NodeIdType samples_num,
                                        NodeType neg_neighbor_type, std::shared_ptr<Tensor> *out) = 0;

  // Node2vec random walk.
  // @param std::vector<NodeIdType> node_list - List of nodes
  // @param std::vector<NodeType> meta_path - node type of each step
  // @param float step_home_param - return hyper parameter in node2vec algorithm
  // @param float step_away_param - in out hyper parameter in node2vec algorithm
  // @param NodeIdType default_node - default node id
  // @param std::shared_ptr<Tensor> *out - Returned nodes id in walk path
  // @return Status The status code returned
  virtual Status RandomWalk(const std::vector<NodeIdType> &node_list, const std::vector<NodeType> &meta_path,
                            float step_home_param, float step_away_param, NodeIdType default_node,
                            std::shared_ptr<Tensor> *out) = 0;

  // Get the feature of a node
  // @param std::shared_ptr<Tensor> nodes - List of nodes
  // @param std::vector<FeatureType> feature_types - Types of features, An error will be reported if the feature type
  // does not exist.
  // @param TensorRow *out - Returned features
  // @return Status The status code returned
  virtual Status GetNodeFeature(const std::shared_ptr<Tensor> &nodes, const std::vector<FeatureType> &feature_types,
                                TensorRow *out) = 0;

  // Get the feature of a edge
  // @param std::shared_ptr<Tensor> edges - List of edges
  // @param std::vector<FeatureType> feature_types - Types of features, An error will be reported if the feature type
  // does not exist.
  // @param Tensor *out - Returned features
  // @return Status The status code returned
  virtual Status GetEdgeFeature(const std::shared_ptr<Tensor> &edges, const std::vector<FeatureType> &feature_types,
                                TensorRow *out) = 0;

  // Return meta information to python layer
  virtual Status GraphInfo(py::dict *out) = 0;

  virtual Status Init() = 0;

  virtual Status Stop() = 0;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_H_
