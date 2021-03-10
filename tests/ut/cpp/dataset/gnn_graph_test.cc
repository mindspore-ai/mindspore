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
#include <map>
#include <memory>
#include <unordered_set>

#include "common/common.h"
#include "gtest/gtest.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/gnn/node.h"
#include "minddata/dataset/engine/gnn/graph_data_impl.h"
#include "minddata/dataset/engine/gnn/graph_loader.h"

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

  using NumNeighborsMap = std::map<NodeIdType, uint32_t>;
  using NodeNeighborsMap = std::map<NodeIdType, NumNeighborsMap>;
  void ParsingNeighbors(const std::shared_ptr<Tensor> &neighbors, NodeNeighborsMap &node_neighbors) {
    auto shape_vec = neighbors->shape().AsVector();
    uint32_t num_members = 1;
    for (size_t i = 1; i < shape_vec.size(); ++i) {
      num_members *= shape_vec[i];
    }
    uint32_t index = 0;
    NodeIdType src_node = 0;
    for (auto node_itr = neighbors->begin<NodeIdType>(); node_itr != neighbors->end<NodeIdType>();
         ++node_itr, ++index) {
      if (index % num_members == 0) {
        src_node = *node_itr;
        continue;
      }
      auto src_node_itr = node_neighbors.find(src_node);
      if (src_node_itr == node_neighbors.end()) {
        node_neighbors[src_node] = {{*node_itr, 1}};
      } else {
        auto nei_itr = src_node_itr->second.find(*node_itr);
        if (nei_itr == src_node_itr->second.end()) {
          src_node_itr->second[*node_itr] = 1;
        } else {
          src_node_itr->second[*node_itr] += 1;
        }
      }
    }
  }

  void CheckNeighborsRatio(const NumNeighborsMap &number_neighbors, const std::vector<WeightType> &weights,
                           float deviation_ratio = 0.2) {
    EXPECT_EQ(number_neighbors.size(), weights.size());
    int index = 0;
    uint32_t pre_num = 0;
    WeightType pre_weight = 1;
    for (auto neighbor : number_neighbors) {
      if (pre_num != 0) {
        float target_ratio = static_cast<float>(pre_weight) / static_cast<float>(weights[index]);
        float current_ratio = static_cast<float>(pre_num) / static_cast<float>(neighbor.second);
        float target_upper = target_ratio * (1 + deviation_ratio);
        float target_lower = target_ratio * (1 - deviation_ratio);
        MS_LOG(INFO) << "current_ratio:" << std::to_string(current_ratio)
                     << " target_upper:" << std::to_string(target_upper)
                     << " target_lower:" << std::to_string(target_lower);
        EXPECT_LE(current_ratio, target_upper);
        EXPECT_GE(current_ratio, target_lower);
      }
      pre_num = neighbor.second;
      pre_weight = weights[index];
      ++index;
    }
  }
};

TEST_F(MindDataTestGNNGraph, TestGetAllNeighbors) {
  std::string path = "data/mindrecord/testGraphData/testdata";
  GraphDataImpl graph(path, 1);
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
  GraphDataImpl graph(path, 1);
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

  TensorRow edge_features;
  s = graph.GetEdgeFeature(edges, meta_info.edge_feature_type, &edge_features);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(edge_features[0]->ToString() ==
              "Tensor (shape: <40>, Type: int32)\n"
              "[0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0]");
  EXPECT_TRUE(edge_features[1]->ToString() ==
              "Tensor (shape: <40>, Type: float32)\n"
              "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2."
              "7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4]");

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
  {
    MS_LOG(INFO) << "Test random sampling.";
    NodeNeighborsMap number_neighbors;
    int count = 0;
    while (count < 1000) {
      neighbors.reset();
      s = graph.GetSampledNeighbors(node_list, {10}, {meta_info.node_type[1]}, SamplingStrategy::kRandom, &neighbors);
      EXPECT_TRUE(s.IsOk());
      EXPECT_TRUE(neighbors->shape().ToString() == "<5,11>");
      ParsingNeighbors(neighbors, number_neighbors);
      ++count;
    }
    CheckNeighborsRatio(number_neighbors[103], {1, 1, 1, 1, 1});
  }

  {
    MS_LOG(INFO) << "Test edge weight sampling.";
    NodeNeighborsMap number_neighbors;
    int count = 0;
    while (count < 1000) {
      neighbors.reset();
      s =
        graph.GetSampledNeighbors(node_list, {10}, {meta_info.node_type[1]}, SamplingStrategy::kEdgeWeight, &neighbors);
      EXPECT_TRUE(s.IsOk());
      EXPECT_TRUE(neighbors->shape().ToString() == "<5,11>");
      ParsingNeighbors(neighbors, number_neighbors);
      ++count;
    }
    CheckNeighborsRatio(number_neighbors[103], {3, 5, 6, 7, 8});
  }

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2, 3}, {meta_info.node_type[1], meta_info.node_type[0]},
                                SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neighbors->shape().ToString() == "<5,9>");

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2, 3, 4},
                                {meta_info.node_type[1], meta_info.node_type[0], meta_info.node_type[1]},
                                SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neighbors->shape().ToString() == "<5,33>");

  neighbors.reset();
  s = graph.GetSampledNeighbors({}, {10}, {meta_info.node_type[1]}, SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.ToString().find("Input node_list is empty.") != std::string::npos);

  neighbors.reset();
  s = graph.GetSampledNeighbors({-1, 1}, {10}, {meta_info.node_type[1]}, SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.ToString().find("Invalid node id") != std::string::npos);

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2, 50}, {meta_info.node_type[0], meta_info.node_type[1]},
                                SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.ToString().find("Wrong samples number") != std::string::npos);

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2}, {5}, SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.ToString().find("Invalid neighbor type") != std::string::npos);

  neighbors.reset();
  s = graph.GetSampledNeighbors(node_list, {2, 3, 4}, {meta_info.node_type[1], meta_info.node_type[0]},
                                SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.ToString().find("The sizes of neighbor_nums and neighbor_types are inconsistent.") !=
              std::string::npos);

  neighbors.reset();
  s = graph.GetSampledNeighbors({301}, {10}, {meta_info.node_type[1]}, SamplingStrategy::kRandom, &neighbors);
  EXPECT_TRUE(s.ToString().find("Invalid node id:301") != std::string::npos);
}

TEST_F(MindDataTestGNNGraph, TestGetNegSampledNeighbors) {
  std::string path = "data/mindrecord/testGraphData/testdata";
  GraphDataImpl graph(path, 1);
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
  s = graph.GetNegSampledNeighbors({-1, 1}, 3, meta_info.node_type[1], &neg_neighbors);
  EXPECT_TRUE(s.ToString().find("Invalid node id") != std::string::npos);

  neg_neighbors.reset();
  s = graph.GetNegSampledNeighbors(node_list, 50, meta_info.node_type[1], &neg_neighbors);
  EXPECT_TRUE(s.ToString().find("Wrong samples number") != std::string::npos);

  neg_neighbors.reset();
  s = graph.GetNegSampledNeighbors(node_list, 3, 3, &neg_neighbors);
  EXPECT_TRUE(s.ToString().find("Invalid neighbor type") != std::string::npos);
}

TEST_F(MindDataTestGNNGraph, TestRandomWalk) {
  std::string path = "data/mindrecord/testGraphData/sns";
  GraphDataImpl graph(path, 1);
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

TEST_F(MindDataTestGNNGraph, TestRandomWalkDefaults) {
  std::string path = "data/mindrecord/testGraphData/sns";
  GraphDataImpl graph(path, 1);
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
  s = graph.RandomWalk(node_list, meta_path, 1.0, 1.0, -1, &walk_path);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(walk_path->shape().ToString() == "<33,60>");
}
