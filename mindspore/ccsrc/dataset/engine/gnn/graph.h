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
#ifndef DATASET_ENGINE_GNN_GRAPH_H_
#define DATASET_ENGINE_GNN_GRAPH_H_

#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/core/tensor_row.h"
#include "dataset/engine/gnn/graph_loader.h"
#include "dataset/engine/gnn/feature.h"
#include "dataset/engine/gnn/node.h"
#include "dataset/engine/gnn/edge.h"
#include "dataset/util/status.h"

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

class Graph {
 public:
  // Constructor
  // @param std::string dataset_file -
  // @param int32_t num_workers - number of parallel threads
  Graph(std::string dataset_file, int32_t num_workers);

  ~Graph() = default;

  // Get all nodes from the graph.
  // @param NodeType node_type - type of node
  // @param std::shared_ptr<Tensor> *out - Returned nodes id
  // @return Status - The error code return
  Status GetAllNodes(NodeType node_type, std::shared_ptr<Tensor> *out);

  // Get all edges from the graph.
  // @param NodeType edge_type - type of edge
  // @param std::shared_ptr<Tensor> *out - Returned edge ids
  // @return Status - The error code return
  Status GetAllEdges(EdgeType edge_type, std::shared_ptr<Tensor> *out);

  // Get the node id from the edge.
  // @param std::vector<EdgeIdType> edge_list - List of edges
  // @param std::shared_ptr<Tensor> *out - Returned node ids
  // @return Status - The error code return
  Status GetNodesFromEdges(const std::vector<EdgeIdType> &edge_list, std::shared_ptr<Tensor> *out);

  // All neighbors of the acquisition node.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param NodeType neighbor_type - The type of neighbor. If the type does not exist, an error will be reported
  // @param std::shared_ptr<Tensor> *out - Returned neighbor's id. Because the number of neighbors at different nodes is
  // different, the returned tensor is output according to the maximum number of neighbors. If the number of neighbors
  // is not enough, fill in tensor as -1.
  // @return Status - The error code return
  Status GetAllNeighbors(const std::vector<NodeIdType> &node_list, NodeType neighbor_type,
                         std::shared_ptr<Tensor> *out);

  // Get sampled neighbors.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param std::vector<NodeIdType> neighbor_nums - Number of neighbors sampled per hop
  // @param std::vector<NodeType> neighbor_types - Neighbor type sampled per hop
  // @param std::shared_ptr<Tensor> *out - Returned neighbor's id.
  // @return Status - The error code return
  Status GetSampledNeighbors(const std::vector<NodeIdType> &node_list, const std::vector<NodeIdType> &neighbor_nums,
                             const std::vector<NodeType> &neighbor_types, std::shared_ptr<Tensor> *out);

  // Get negative sampled neighbors.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param NodeIdType samples_num - Number of neighbors sampled
  // @param NodeType neg_neighbor_type - The type of negative neighbor.
  // @param std::shared_ptr<Tensor> *out - Returned negative neighbor's id.
  // @return Status - The error code return
  Status GetNegSampledNeighbors(const std::vector<NodeIdType> &node_list, NodeIdType samples_num,
                                NodeType neg_neighbor_type, std::shared_ptr<Tensor> *out);

  Status RandomWalk(const std::vector<NodeIdType> &node_list, const std::vector<NodeType> &meta_path, float p, float q,
                    NodeIdType default_node, std::shared_ptr<Tensor> *out);

  // Get the feature of a node
  // @param std::shared_ptr<Tensor> nodes - List of nodes
  // @param std::vector<FeatureType> feature_types - Types of features, An error will be reported if the feature type
  // does not exist.
  // @param TensorRow *out - Returned features
  // @return Status - The error code return
  Status GetNodeFeature(const std::shared_ptr<Tensor> &nodes, const std::vector<FeatureType> &feature_types,
                        TensorRow *out);

  // Get the feature of a edge
  // @param std::shared_ptr<Tensor> edget - List of edges
  // @param std::vector<FeatureType> feature_types - Types of features, An error will be reported if the feature type
  // does not exist.
  // @param Tensor *out - Returned features
  // @return Status - The error code return
  Status GetEdgeFeature(const std::shared_ptr<Tensor> &edget, const std::vector<FeatureType> &feature_types,
                        TensorRow *out);

  // Get meta information of graph
  // @param MetaInfo *meta_info - Returned meta information
  // @return Status - The error code return
  Status GetMetaInfo(MetaInfo *meta_info);

  // Return meta information to python layer
  Status GraphInfo(py::dict *out);

  Status Init();

 private:
  // Load graph data from mindrecord file
  // @return Status - The error code return
  Status LoadNodeAndEdge();

  // Create Tensor By Vector
  // @param std::vector<std::vector<T>> &data -
  // @param DataType type -
  // @param std::shared_ptr<Tensor> *out -
  // @return Status - The error code return
  template <typename T>
  Status CreateTensorByVector(const std::vector<std::vector<T>> &data, DataType type, std::shared_ptr<Tensor> *out);

  // Complete vector
  // @param std::vector<std::vector<T>> *data - To be completed vector
  // @param size_t max_size - The size of the completed vector
  // @param T default_value - Filled default
  // @return Status - The error code return
  template <typename T>
  Status ComplementVector(std::vector<std::vector<T>> *data, size_t max_size, T default_value);

  // Get the default feature of a node
  // @param FeatureType feature_type -
  // @param std::shared_ptr<Feature> *out_feature - Returned feature
  // @return Status - The error code return
  Status GetNodeDefaultFeature(FeatureType feature_type, std::shared_ptr<Feature> *out_feature);

  // Find node object using node id
  // @param NodeIdType id -
  // @param std::shared_ptr<Node> *node - Returned node object
  // @return Status - The error code return
  Status GetNodeByNodeId(NodeIdType id, std::shared_ptr<Node> *node);

  // Negative sampling
  // @param std::vector<NodeIdType> &input_data - The data set to be sampled
  // @param std::unordered_set<NodeIdType> &exclude_data - Data to be excluded
  // @param int32_t samples_num -
  // @param std::vector<NodeIdType> *out_samples - Sampling results returned
  // @return Status - The error code return
  Status NegativeSample(const std::vector<NodeIdType> &input_data, const std::unordered_set<NodeIdType> &exclude_data,
                        int32_t samples_num, std::vector<NodeIdType> *out_samples);

  std::string dataset_file_;
  int32_t num_workers_;  // The number of worker threads
  std::mt19937 rnd_;

  std::unordered_map<NodeType, std::vector<NodeIdType>> node_type_map_;
  std::unordered_map<NodeIdType, std::shared_ptr<Node>> node_id_map_;

  std::unordered_map<EdgeType, std::vector<EdgeIdType>> edge_type_map_;
  std::unordered_map<EdgeIdType, std::shared_ptr<Edge>> edge_id_map_;

  std::unordered_map<NodeType, std::unordered_set<FeatureType>> node_feature_map_;
  std::unordered_map<EdgeType, std::unordered_set<FeatureType>> edge_feature_map_;

  std::unordered_map<FeatureType, std::shared_ptr<Feature>> default_feature_map_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_GNN_GRAPH_H_
