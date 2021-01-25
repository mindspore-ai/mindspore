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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_LOADER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_LOADER_H_

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/gnn/edge.h"
#include "minddata/dataset/engine/gnn/feature.h"
#include "minddata/dataset/engine/gnn/graph_feature_parser.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "minddata/dataset/engine/gnn/graph_shared_memory.h"
#endif
#include "minddata/dataset/engine/gnn/node.h"
#include "minddata/dataset/util/status.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/dataset/engine/gnn/graph_data_impl.h"
namespace mindspore {
namespace dataset {
namespace gnn {

using mindrecord::ShardReader;
using NodeIdMap = std::unordered_map<NodeIdType, std::shared_ptr<Node>>;
using EdgeIdMap = std::unordered_map<EdgeIdType, std::shared_ptr<Edge>>;
using NodeTypeMap = std::unordered_map<NodeType, std::vector<NodeIdType>>;
using EdgeTypeMap = std::unordered_map<EdgeType, std::vector<EdgeIdType>>;
using NodeFeatureMap = std::unordered_map<NodeType, std::unordered_set<FeatureType>>;
using EdgeFeatureMap = std::unordered_map<EdgeType, std::unordered_set<FeatureType>>;
using DefaultNodeFeatureMap = std::unordered_map<FeatureType, std::shared_ptr<Feature>>;
using DefaultEdgeFeatureMap = std::unordered_map<FeatureType, std::shared_ptr<Feature>>;

// this class interfaces with the underlying storage format (mindrecord)
// it returns raw nodes and edges via GetNodesAndEdges
// it is then the responsibility of graph to construct itself based on the nodes and edges
// if needed, this class could become a base where each derived class handles a specific storage format
class GraphLoader {
 public:
  GraphLoader(GraphDataImpl *graph_impl, std::string mr_filepath, int32_t num_workers = 4, bool server_mode = false);

  ~GraphLoader() = default;
  // Init mindrecord and load everything into memory multi-threaded
  // @return Status - the status code
  Status InitAndLoad();

  // this function will query mindrecord and construct all nodes and edges
  // nodes and edges are added to map without any connection. That's because there nodes and edges are read in
  // random order. src_node and dst_node in Edge are node_id only with -1 as type.
  // features attached to each node and edge are expected to be filled correctly
  Status GetNodesAndEdges();

 private:
  //
  // worker thread that reads mindrecord file
  // @param int32_t worker_id - id of each worker
  // @return Status - the status code
  Status WorkerEntry(int32_t worker_id);

  // Load a node based on 1 row of mindrecord, returns a shared_ptr<Node>
  // @param std::vector<uint8_t> &blob - contains data in blob field in mindrecord
  // @param mindrecord::json &jsn - contains raw data
  // @param std::shared_ptr<Node> *node - return value
  // @param NodeFeatureMap *feature_map -
  // @param DefaultNodeFeatureMap *default_feature -
  // @return Status - the status code
  Status LoadNode(const std::vector<uint8_t> &blob, const mindrecord::json &jsn, std::shared_ptr<Node> *node,
                  NodeFeatureMap *feature_map, DefaultNodeFeatureMap *default_feature);

  // @param std::vector<uint8_t> &blob - contains data in blob field in mindrecord
  // @param mindrecord::json &jsn - contains raw data
  // @param std::shared_ptr<Edge> *edge - return value, the edge ptr, edge is not yet connected
  // @param FeatureMap *feature_map
  // @param DefaultEdgeFeatureMap *default_feature -
  // @return Status - the status code
  Status LoadEdge(const std::vector<uint8_t> &blob, const mindrecord::json &jsn, std::shared_ptr<Edge> *edge,
                  EdgeFeatureMap *feature_map, DefaultEdgeFeatureMap *default_feature);

  // merge NodeFeatureMap and EdgeFeatureMap of each worker into 1
  void MergeFeatureMaps();

  GraphDataImpl *graph_impl_;
  std::string mr_path_;
  const int32_t num_workers_;
  std::atomic_int row_id_;
  std::unique_ptr<ShardReader> shard_reader_;
  std::unique_ptr<GraphFeatureParser> graph_feature_parser_;
  std::vector<std::deque<std::shared_ptr<Node>>> n_deques_;
  std::vector<std::deque<std::shared_ptr<Edge>>> e_deques_;
  std::vector<NodeFeatureMap> n_feature_maps_;
  std::vector<EdgeFeatureMap> e_feature_maps_;
  std::vector<DefaultNodeFeatureMap> default_node_feature_maps_;
  std::vector<DefaultEdgeFeatureMap> default_edge_feature_maps_;
  const std::vector<std::string> required_key_;
  std::unordered_map<std::string, bool> optional_key_;
};
}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_GNN_GRAPH_LOADER_H_
