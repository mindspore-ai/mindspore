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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/engine/gnn/graph_loader.h"
#include "dataset/engine/gnn/feature.h"
#include "dataset/engine/gnn/node.h"
#include "dataset/engine/gnn/edge.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
namespace gnn {

struct NodeMetaInfo {
  NodeType type;
  NodeIdType num;
  std::vector<FeatureType> feature_type;
  NodeMetaInfo() {
    type = 0;
    num = 0;
  }
};

struct EdgeMetaInfo {
  EdgeType type;
  EdgeIdType num;
  std::vector<FeatureType> feature_type;
  EdgeMetaInfo() {
    type = 0;
    num = 0;
  }
};

class Graph {
 public:
  // Constructor
  // @param std::string dataset_file -
  // @param int32_t num_workers - number of parallel threads
  Graph(std::string dataset_file, int32_t num_workers);

  ~Graph() = default;

  // Get the nodes from the graph.
  // @param NodeType node_type - type of node
  // @param NodeIdType node_num - Number of nodes to be acquired, if -1 means all nodes are acquired
  // @param std::shared_ptr<Tensor> *out - Returned nodes id
  // @return Status - The error code return
  Status GetNodes(NodeType node_type, NodeIdType node_num, std::shared_ptr<Tensor> *out);

  // Get the edges from the graph.
  // @param NodeType edge_type - type of edge
  // @param NodeIdType edge_num - Number of edges to be acquired, if -1 means all edges are acquired
  // @param std::shared_ptr<Tensor> *out - Returned edge ids
  // @return Status - The error code return
  Status GetEdges(EdgeType edge_type, EdgeIdType edge_num, std::shared_ptr<Tensor> *out);

  // All neighbors of the acquisition node.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param NodeType neighbor_type - The type of neighbor. If the type does not exist, an error will be reported
  // @param std::shared_ptr<Tensor> *out - Returned neighbor's id. Because the number of neighbors at different nodes is
  // different, the returned tensor is output according to the maximum number of neighbors. If the number of neighbors
  // is not enough, fill in tensor as -1.
  // @return Status - The error code return
  Status GetAllNeighbors(const std::vector<NodeIdType> &node_list, NodeType neighbor_type,
                         std::shared_ptr<Tensor> *out);

  Status GetSampledNeighbor(const std::vector<NodeIdType> &node_list, const std::vector<NodeIdType> &neighbor_nums,
                            const std::vector<NodeType> &neighbor_types, std::shared_ptr<Tensor> *out);
  Status GetNegSampledNeighbor(const std::vector<NodeIdType> &node_list, NodeIdType samples_num,
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
  // @param std::vector<NodeMetaInfo> *node_info - Returned meta information of node
  // @param std::vector<NodeMetaInfo> *node_info - Returned meta information of edge
  // @return Status - The error code return
  Status GetMetaInfo(std::vector<NodeMetaInfo> *node_info, std::vector<EdgeMetaInfo> *edge_info);

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

  std::string dataset_file_;
  int32_t num_workers_;  // The number of worker threads

  std::unordered_map<NodeType, std::vector<NodeIdType>> node_type_map_;
  std::unordered_map<NodeIdType, std::shared_ptr<Node>> node_id_map_;

  std::unordered_map<EdgeType, std::vector<EdgeIdType>> edge_type_map_;
  std::unordered_map<EdgeIdType, std::shared_ptr<Edge>> edge_id_map_;

  std::unordered_map<NodeType, std::unordered_set<FeatureType>> node_feature_map_;
  std::unordered_map<NodeType, std::unordered_set<FeatureType>> edge_feature_map_;

  std::unordered_map<FeatureType, std::shared_ptr<Feature>> default_feature_map_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_ENGINE_GNN_GRAPH_H_
