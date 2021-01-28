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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_CLIENT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_CLIENT_H_

#include <algorithm>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>

#if !defined(_WIN32) && !defined(_WIN64)
#include "proto/gnn_graph_data.grpc.pb.h"
#include "proto/gnn_graph_data.pb.h"
#endif
#include "minddata/dataset/engine/gnn/graph_data.h"
#include "minddata/dataset/engine/gnn/graph_feature_parser.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "minddata/dataset/engine/gnn/graph_shared_memory.h"
#endif
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_column.h"

namespace mindspore {
namespace dataset {
namespace gnn {

class GraphDataClient : public GraphData {
 public:
  // Constructor
  // @param std::string dataset_file -
  // @param int32_t num_workers - number of parallel threads
  GraphDataClient(const std::string &dataset_file, const std::string &hostname, int32_t port);

  ~GraphDataClient();

  Status Init() override;

  Status Stop() override;

  // Get all nodes from the graph.
  // @param NodeType node_type - type of node
  // @param std::shared_ptr<Tensor> *out - Returned nodes id
  // @return Status The status code returned
  Status GetAllNodes(NodeType node_type, std::shared_ptr<Tensor> *out) override;

  // Get all edges from the graph.
  // @param NodeType edge_type - type of edge
  // @param std::shared_ptr<Tensor> *out - Returned edge ids
  // @return Status The status code returned
  Status GetAllEdges(EdgeType edge_type, std::shared_ptr<Tensor> *out) override;

  // Get the node id from the edge.
  // @param std::vector<EdgeIdType> edge_list - List of edges
  // @param std::shared_ptr<Tensor> *out - Returned node ids
  // @return Status The status code returned
  Status GetNodesFromEdges(const std::vector<EdgeIdType> &edge_list, std::shared_ptr<Tensor> *out) override;

  // All neighbors of the acquisition node.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param NodeType neighbor_type - The type of neighbor. If the type does not exist, an error will be reported
  // @param std::shared_ptr<Tensor> *out - Returned neighbor's id. Because the number of neighbors at different nodes is
  // different, the returned tensor is output according to the maximum number of neighbors. If the number of neighbors
  // is not enough, fill in tensor as -1.
  // @return Status The status code returned
  Status GetAllNeighbors(const std::vector<NodeIdType> &node_list, NodeType neighbor_type,
                         std::shared_ptr<Tensor> *out) override;

  // Get sampled neighbors.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param std::vector<NodeIdType> neighbor_nums - Number of neighbors sampled per hop
  // @param std::vector<NodeType> neighbor_types - Neighbor type sampled per hop
  // @param std::SamplingStrategy strategy - Sampling strategy
  // @param std::shared_ptr<Tensor> *out - Returned neighbor's id.
  // @return Status The status code returned
  Status GetSampledNeighbors(const std::vector<NodeIdType> &node_list, const std::vector<NodeIdType> &neighbor_nums,
                             const std::vector<NodeType> &neighbor_types, SamplingStrategy strategy,
                             std::shared_ptr<Tensor> *out) override;

  // Get negative sampled neighbors.
  // @param std::vector<NodeType> node_list - List of nodes
  // @param NodeIdType samples_num - Number of neighbors sampled
  // @param NodeType neg_neighbor_type - The type of negative neighbor.
  // @param std::shared_ptr<Tensor> *out - Returned negative neighbor's id.
  // @return Status The status code returned
  Status GetNegSampledNeighbors(const std::vector<NodeIdType> &node_list, NodeIdType samples_num,
                                NodeType neg_neighbor_type, std::shared_ptr<Tensor> *out) override;

  // Node2vec random walk.
  // @param std::vector<NodeIdType> node_list - List of nodes
  // @param std::vector<NodeType> meta_path - node type of each step
  // @param float step_home_param - return hyper parameter in node2vec algorithm
  // @param float step_away_param - in out hyper parameter in node2vec algorithm
  // @param NodeIdType default_node - default node id
  // @param std::shared_ptr<Tensor> *out - Returned nodes id in walk path
  // @return Status The status code returned
  Status RandomWalk(const std::vector<NodeIdType> &node_list, const std::vector<NodeType> &meta_path,
                    float step_home_param, float step_away_param, NodeIdType default_node,
                    std::shared_ptr<Tensor> *out) override;

  // Get the feature of a node
  // @param std::shared_ptr<Tensor> nodes - List of nodes
  // @param std::vector<FeatureType> feature_types - Types of features, An error will be reported if the feature type
  // does not exist.
  // @param TensorRow *out - Returned features
  // @return Status The status code returned
  Status GetNodeFeature(const std::shared_ptr<Tensor> &nodes, const std::vector<FeatureType> &feature_types,
                        TensorRow *out) override;

  // Get the feature of a edge
  // @param std::shared_ptr<Tensor> edges - List of edges
  // @param std::vector<FeatureType> feature_types - Types of features, An error will be reported if the feature type
  // does not exist.
  // @param Tensor *out - Returned features
  // @return Status The status code returned
  Status GetEdgeFeature(const std::shared_ptr<Tensor> &edges, const std::vector<FeatureType> &feature_types,
                        TensorRow *out) override;

  // Return meta information to python layer
  Status GraphInfo(py::dict *out) override;

 private:
#if !defined(_WIN32) && !defined(_WIN64)
  Status ParseNodeFeatureFromMemory(const std::shared_ptr<Tensor> &nodes, FeatureType feature_type,
                                    const std::shared_ptr<Tensor> &memory_tensor, std::shared_ptr<Tensor> *out);

  Status ParseEdgeFeatureFromMemory(const std::shared_ptr<Tensor> &edges, FeatureType feature_type,
                                    const std::shared_ptr<Tensor> &memory_tensor, std::shared_ptr<Tensor> *out);

  Status GetNodeDefaultFeature(FeatureType feature_type, std::shared_ptr<Tensor> *out_feature);

  Status GetEdgeDefaultFeature(FeatureType feature_type, std::shared_ptr<Tensor> *out_feature);

  Status GetGraphData(const GnnGraphDataRequestPb &request, GnnGraphDataResponsePb *response);

  Status GetGraphDataTensor(const GnnGraphDataRequestPb &request, GnnGraphDataResponsePb *response,
                            std::shared_ptr<Tensor> *out);

  Status RegisterToServer();

  Status UnRegisterToServer();

  Status InitFeatureParser();

  Status CheckPid() {
    CHECK_FAIL_RETURN_UNEXPECTED(pid_ == getpid(),
                                 "Multi-process mode is not supported, please change to use multi-thread");
    return Status::OK();
  }
#endif

  std::string dataset_file_;
  std::string host_;
  int32_t port_;
  int32_t pid_;
  mindrecord::json data_schema_;
#if !defined(_WIN32) && !defined(_WIN64)
  std::unique_ptr<GnnGraphData::Stub> stub_;
  key_t shared_memory_key_;
  int64_t shared_memory_size_;
  std::unique_ptr<GraphFeatureParser> graph_feature_parser_;
  std::unique_ptr<GraphSharedMemory> graph_shared_memory_;
  std::unordered_map<FeatureType, std::shared_ptr<Tensor>> default_node_feature_map_;
  std::unordered_map<FeatureType, std::shared_ptr<Tensor>> default_edge_feature_map_;
#endif
  bool registered_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_CLIENT_H_
