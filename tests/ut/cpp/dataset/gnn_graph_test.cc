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
#include <algorithm>
#include <string>
#include <memory>
#include <unordered_set>

#include "common/common.h"
#include "gtest/gtest.h"
#include "dataset/util/status.h"
#include "dataset/engine/gnn/node.h"
#include "dataset/engine/gnn/graph_loader.h"

using namespace mindspore::dataset;
using namespace mindspore::dataset::gnn;

#define print_int_vec(_i, _str)                                           \
  do {                                                                    \
    std::stringstream ss;                                                 \
    std::copy(_i.begin(), _i.end(), std::ostream_iterator<int>(ss, " ")); \
    MS_LOG(INFO) << _str << " " << ss.str();                              \
  } while (false)

class MindDataTestGNNGraph : public UT::Common {
 protected:
  MindDataTestGNNGraph() = default;
};

TEST_F(MindDataTestGNNGraph, TestGraphLoader) {
  std::string path = "data/mindrecord/testGraphData/testdata";
  GraphLoader gl(path, 4);
  EXPECT_TRUE(gl.InitAndLoad().IsOk());
  NodeIdMap n_id_map;
  EdgeIdMap e_id_map;
  NodeTypeMap n_type_map;
  EdgeTypeMap e_type_map;
  NodeFeatureMap n_feature_map;
  EdgeFeatureMap e_feature_map;
  DefaultFeatureMap default_feature_map;
  EXPECT_TRUE(gl.GetNodesAndEdges(&n_id_map, &e_id_map, &n_type_map, &e_type_map, &n_feature_map, &e_feature_map,
                                  &default_feature_map)
                .IsOk());
  EXPECT_EQ(n_id_map.size(), 20);
  EXPECT_EQ(e_id_map.size(), 40);
  EXPECT_EQ(n_type_map[2].size(), 10);
  EXPECT_EQ(n_type_map[1].size(), 10);
}

TEST_F(MindDataTestGNNGraph, TestGetAllNeighbors) {
  std::string path = "data/mindrecord/testGraphData/testdata";
  Graph graph(path, 1);
  Status s = graph.Init();
  EXPECT_TRUE(s.IsOk());

  MetaInfo meta_info;
  s = graph.GetMetaInfo(&meta_info);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(meta_info.node_type.size() == 2);

  std::shared_ptr<Tensor> nodes;
  s = graph.GetAllNodes(meta_info.node_type[0], &nodes);
  EXPECT_TRUE(s.IsOk());
  std::vector<NodeIdType> node_list;
  for (auto itr = nodes->begin<NodeIdType>(); itr != nodes->end<NodeIdType>(); ++itr) {
    node_list.push_back(*itr);
    if (node_list.size() >= 10) {
      break;
    }
  }
  std::shared_ptr<Tensor> neighbors;
  s = graph.GetAllNeighbors(node_list, meta_info.node_type[1], &neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neighbors->shape().ToString() == "<10,6>");
  TensorRow features;
  s = graph.GetNodeFeature(nodes, meta_info.node_feature_type, &features);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(features.size() == 4);
  EXPECT_TRUE(features[0]->shape().ToString() == "<10,5>");
  EXPECT_TRUE(features[0]->ToString() ==
              "Tensor (shape: <10,5>, Type: int32)\n"
              "[[0,1,0,0,0],[1,0,0,0,1],[0,0,1,1,0],[0,0,0,0,0],[1,1,0,1,0],[0,0,0,0,1],[0,1,0,0,0],[0,0,0,1,1],[0,1,1,"
              "0,0],[0,1,0,1,0]]");
  EXPECT_TRUE(features[1]->shape().ToString() == "<10>");
  EXPECT_TRUE(features[1]->ToString() ==
              "Tensor (shape: <10>, Type: float32)\n[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]");
  EXPECT_TRUE(features[2]->shape().ToString() == "<10>");
  EXPECT_TRUE(features[2]->ToString() == "Tensor (shape: <10>, Type: int32)\n[1,2,3,1,4,3,5,3,5,4]");
}

TEST_F(MindDataTestGNNGraph, TestGetSampledNeighbors) {
  std::string path = "data/mindrecord/testGraphData/testdata";
  Graph graph(path, 1);
  Status s = graph.Init();
  EXPECT_TRUE(s.IsOk());

  MetaInfo meta_info;
  s = graph.GetMetaInfo(&meta_info);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(meta_info.node_type.size() == 2);

  std::shared_ptr<Tensor> edges;
  s = graph.GetAllEdges(meta_info.edge_type[0], &edges);
  EXPECT_TRUE(s.IsOk());
  std::vector<EdgeIdType> edge_list;
  edge_list.resize(edges->Size());
  std::transform(edges->begin<EdgeIdType>(), edges->end<EdgeIdType>(), edge_list.begin(),
                 [](const EdgeIdType edge) { return edge; });

  std::shared_ptr<Tensor> nodes;
  s = graph.GetNodesFromEdges(edge_list, &nodes);
  EXPECT_TRUE(s.IsOk());
  std::unordered_set<NodeIdType> node_set;
  std::vector<NodeIdType> node_list;
  int index = 0;
  for (auto itr = nodes->begin<NodeIdType>(); itr != nodes->end<NodeIdType>(); ++itr) {
    index++;
    if (index % 2 == 0) {
      continue;
    }
    node_set.emplace(*itr);
    if (node_set.size() >= 5) {
      break;
    }
  }
  node_list.resize(node_set.size());
  std::transform(node_set.begin(), node_set.end(), node_list.begin(), [](const NodeIdType node) { return node; });

  std::shared_ptr<Tensor> neighbors;
  s = graph.GetSampledNeighbors(node_list, {10}, {meta_info.node_type[1]}, &neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neighbors->shape().ToString() == "<5,11>");

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2, 3}, {meta_info.node_type[1], meta_info.node_type[0]}, &neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neighbors->shape().ToString() == "<5,9>");

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2, 3, 4},
                                {meta_info.node_type[1], meta_info.node_type[0], meta_info.node_type[1]}, &neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neighbors->shape().ToString() == "<5,33>");

  neighbors.reset();
  s = graph.GetSampledNeighbors({}, {10}, {meta_info.node_type[1]}, &neighbors);
  EXPECT_TRUE(s.ToString().find("Input node_list is empty.") != std::string::npos);

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2, 3, 4}, {meta_info.node_type[1], meta_info.node_type[0]}, &neighbors);
  EXPECT_TRUE(s.ToString().find("The sizes of neighbor_nums and neighbor_types are inconsistent.") !=
              std::string::npos);

  neighbors.reset();
  s = graph.GetSampledNeighbors({301}, {10}, {meta_info.node_type[1]}, &neighbors);
  EXPECT_TRUE(s.ToString().find("Invalid node id:301") != std::string::npos);
}

TEST_F(MindDataTestGNNGraph, TestGetNegSampledNeighbors) {
  std::string path = "data/mindrecord/testGraphData/testdata";
  Graph graph(path, 1);
  Status s = graph.Init();
  EXPECT_TRUE(s.IsOk());

  MetaInfo meta_info;
  s = graph.GetMetaInfo(&meta_info);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(meta_info.node_type.size() == 2);

  std::shared_ptr<Tensor> nodes;
  s = graph.GetAllNodes(meta_info.node_type[0], &nodes);
  EXPECT_TRUE(s.IsOk());
  std::vector<NodeIdType> node_list;
  for (auto itr = nodes->begin<NodeIdType>(); itr != nodes->end<NodeIdType>(); ++itr) {
    node_list.push_back(*itr);
    if (node_list.size() >= 10) {
      break;
    }
  }
  std::shared_ptr<Tensor> neg_neighbors;
  s = graph.GetNegSampledNeighbors(node_list, 3, meta_info.node_type[1], &neg_neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neg_neighbors->shape().ToString() == "<10,4>");

  neg_neighbors.reset();
  s = graph.GetNegSampledNeighbors({}, 3, meta_info.node_type[1], &neg_neighbors);
  EXPECT_TRUE(s.ToString().find("Input node_list is empty.") != std::string::npos);

  neg_neighbors.reset();
  s = graph.GetNegSampledNeighbors(node_list, 3, 3, &neg_neighbors);
  EXPECT_TRUE(s.ToString().find("Invalid node type:3") != std::string::npos);
}

TEST_F(MindDataTestGNNGraph, TestRandomWalk) {
  std::string path = "data/mindrecord/testGraphData/sns";
  Graph graph(path, 1);
  Status s = graph.Init();
  EXPECT_TRUE(s.IsOk());

  MetaInfo meta_info;
  s = graph.GetMetaInfo(&meta_info);
  EXPECT_TRUE(s.IsOk());

  std::shared_ptr<Tensor> nodes;
  s = graph.GetAllNodes(meta_info.node_type[0], &nodes);
  EXPECT_TRUE(s.IsOk());
  std::vector<NodeIdType> node_list;
  for (auto itr = nodes->begin<NodeIdType>(); itr != nodes->end<NodeIdType>(); ++itr) {
    node_list.push_back(*itr);
  }

  print_int_vec(node_list, "node list ");
  std::vector<NodeType> meta_path(59, 1);
  std::shared_ptr<Tensor> walk_path;
  s = graph.RandomWalk(node_list, meta_path, 2.0, 0.5, -1, &walk_path);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(walk_path->shape().ToString() == "<33,60>");
}