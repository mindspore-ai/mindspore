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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_IMPL_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>

#include "minddata/dataset/engine/gnn/graph_data.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "minddata/dataset/engine/gnn/graph_shared_memory.h"
#endif
#include "minddata/mindrecord/include/common/shard_utils.h"

namespace mindspore {
namespace dataset {
namespace gnn {

const float kGnnEpsilon = 0.0001;
const uint32_t kMaxNumWalks = 80;
using StochasticIndex = std::pair<std::vector<int32_t>, std::vector<float>>;

class GraphDataImpl : public GraphData {
 public:
  // Constructor
  // @param std::string dataset_file -
  // @param int32_t num_workers - number of parallel threads
  GraphDataImpl(std::string dataset_file, int32_t num_workers, bool server_mode = false);

  ~GraphDataImpl();

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

  Status GetNodeFeatureSharedMemory(const std::shared_ptr<Tensor> &nodes, FeatureType type,
                                    std::shared_ptr<Tensor> *out);

  // Get the feature of a edge
  // @param std::shared_ptr<Tensor> edges - List of edges
  // @param std::vector<FeatureType> feature_types - Types of features, An error will be reported if the feature type
  // does not exist.
  // @param Tensor *out - Returned features
  // @return Status The status code returned
  Status GetEdgeFeature(const std::shared_ptr<Tensor> &edges, const std::vector<FeatureType> &feature_types,
                        TensorRow *out) override;

  Status GetEdgeFeatureSharedMemory(const std::shared_ptr<Tensor> &edges, FeatureType type,
                                    std::shared_ptr<Tensor> *out);

  // Get meta information of graph
  // @param MetaInfo *meta_info - Returned meta information
  // @return Status The status code returned
  Status GetMetaInfo(MetaInfo *meta_info);

#ifdef ENABLE_PYTHON
  // Return meta information to python layer
  Status GraphInfo(py::dict *out) override;
#endif

  const std::unordered_map<FeatureType, std::shared_ptr<Feature>> *GetAllDefaultNodeFeatures() {
    return &default_node_feature_map_;
  }

  const std::unordered_map<FeatureType, std::shared_ptr<Feature>> *GetAllDefaultEdgeFeatures() {
    return &default_edge_feature_map_;
  }

  Status Init() override;

  Status Stop() override { return Status::OK(); }

  std::string GetDataSchema() { return data_schema_.dump(); }

#if !defined(_WIN32) && !defined(_WIN64)
  key_t GetSharedMemoryKey() { return graph_shared_memory_->memory_key(); }

  int64_t GetSharedMemorySize() { return graph_shared_memory_->memory_size(); }
#endif

 private:
  friend class GraphLoader;
  class RandomWalkBase {
   public:
    explicit RandomWalkBase(GraphDataImpl *graph);

    Status Build(const std::vector<NodeIdType> &node_list, const std::vector<NodeType> &meta_path,
                 float step_home_param = 1.0, float step_away_param = 1.0, NodeIdType default_node = -1,
                 int32_t num_walks = 1, int32_t num_workers = 1);

    ~RandomWalkBase() = default;

    Status SimulateWalk(std::vector<std::vector<NodeIdType>> *walks);

   private:
    Status Node2vecWalk(const NodeIdType &start_node, std::vector<NodeIdType> *walk_path);

    Status GetNodeProbability(const NodeIdType &node_id, const NodeType &node_type,
                              std::shared_ptr<StochasticIndex> *node_probability);

    Status GetEdgeProbability(const NodeIdType &src, const NodeIdType &dst, uint32_t meta_path_index,
                              std::shared_ptr<StochasticIndex> *edge_probability);

    static StochasticIndex GenerateProbability(const std::vector<float> &probability);

    static uint32_t WalkToNextNode(const StochasticIndex &stochastic_index);

    template <typename T>
    std::vector<float> Normalize(const std::vector<T> &non_normalized_probability);

    GraphDataImpl *graph_;
    std::vector<NodeIdType> node_list_;
    std::vector<NodeType> meta_path_;
    float step_home_param_;  // Return hyper parameter. Default is 1.0
    float step_away_param_;  // In out hyper parameter. Default is 1.0
    NodeIdType default_node_;

    int32_t num_walks_;    // Number of walks per source. Default is 1
    int32_t num_workers_;  // The number of worker threads. Default is 1
  };

  // Load graph data from mindrecord file
  // @return Status The status code returned
  Status LoadNodeAndEdge();

  // Create Tensor By Vector
  // @param std::vector<std::vector<T>> &data -
  // @param DataType type -
  // @param std::shared_ptr<Tensor> *out -
  // @return Status The status code returned
  template <typename T>
  Status CreateTensorByVector(const std::vector<std::vector<T>> &data, DataType type, std::shared_ptr<Tensor> *out);

  // Complete vector
  // @param std::vector<std::vector<T>> *data - To be completed vector
  // @param size_t max_size - The size of the completed vector
  // @param T default_value - Filled default
  // @return Status The status code returned
  template <typename T>
  Status ComplementVector(std::vector<std::vector<T>> *data, size_t max_size, T default_value);

  // Get the default feature of a node
  // @param FeatureType feature_type -
  // @param std::shared_ptr<Feature> *out_feature - Returned feature
  // @return Status The status code returned
  Status GetNodeDefaultFeature(FeatureType feature_type, std::shared_ptr<Feature> *out_feature);

  // Get the default feature of a edge
  // @param FeatureType feature_type -
  // @param std::shared_ptr<Feature> *out_feature - Returned feature
  // @return Status The status code returned
  Status GetEdgeDefaultFeature(FeatureType feature_type, std::shared_ptr<Feature> *out_feature);

  // Find node object using node id
  // @param NodeIdType id -
  // @param std::shared_ptr<Node> *node - Returned node object
  // @return Status The status code returned
  Status GetNodeByNodeId(NodeIdType id, std::shared_ptr<Node> *node);

  // Find edge object using edge id
  // @param EdgeIdType id -
  // @param std::shared_ptr<Node> *edge - Returned edge object
  // @return Status The status code returned
  Status GetEdgeByEdgeId(EdgeIdType id, std::shared_ptr<Edge> *edge);

  // Negative sampling
  // @param std::vector<NodeIdType> &input_data - The data set to be sampled
  // @param std::unordered_set<NodeIdType> &exclude_data - Data to be excluded
  // @param int32_t samples_num -
  // @param std::vector<NodeIdType> *out_samples - Sampling results returned
  // @return Status The status code returned
  Status NegativeSample(const std::vector<NodeIdType> &data, const std::vector<NodeIdType> shuffled_ids,
                        size_t *start_index, const std::unordered_set<NodeIdType> &exclude_data, int32_t samples_num,
                        std::vector<NodeIdType> *out_samples);

  Status CheckSamplesNum(NodeIdType samples_num);

  Status CheckNeighborType(NodeType neighbor_type);

  std::string dataset_file_;
  int32_t num_workers_;  // The number of worker threads
  std::mt19937 rnd_;
  RandomWalkBase random_walk_;
  mindrecord::json data_schema_;
  bool server_mode_;
#if !defined(_WIN32) && !defined(_WIN64)
  std::unique_ptr<GraphSharedMemory> graph_shared_memory_;
#endif
  std::unordered_map<NodeType, std::vector<NodeIdType>> node_type_map_;
  std::unordered_map<NodeIdType, std::shared_ptr<Node>> node_id_map_;

  std::unordered_map<EdgeType, std::vector<EdgeIdType>> edge_type_map_;
  std::unordered_map<EdgeIdType, std::shared_ptr<Edge>> edge_id_map_;

  std::unordered_map<NodeType, std::unordered_set<FeatureType>> node_feature_map_;
  std::unordered_map<EdgeType, std::unordered_set<FeatureType>> edge_feature_map_;

  std::unordered_map<FeatureType, std::shared_ptr<Feature>> default_node_feature_map_;
  std::unordered_map<FeatureType, std::shared_ptr<Feature>> default_edge_feature_map_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_DATA_IMPL_H_
