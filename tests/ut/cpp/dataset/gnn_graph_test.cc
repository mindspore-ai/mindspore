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
#include <string>
#include <memory>

#include "common/common.h"
#include "gtest/gtest.h"
#include "dataset/util/status.h"
#include "dataset/engine/gnn/node.h"
#include "dataset/engine/gnn/graph_loader.h"

using namespace mindspore::dataset;
using namespace mindspore::dataset::gnn;

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
  EXPECT_EQ(e_id_map.size(), 20);
  EXPECT_EQ(n_type_map[2].size(), 10);
  EXPECT_EQ(n_type_map[1].size(), 10);
}

TEST_F(MindDataTestGNNGraph, TestGetAllNeighbors) {
  std::string path = "data/mindrecord/testGraphData/testdata";
  Graph graph(path, 1);
  Status s = graph.Init();
  EXPECT_TRUE(s.IsOk());

  std::vector<NodeMetaInfo> node_info;
  std::vector<EdgeMetaInfo> edge_info;
  s = graph.GetMetaInfo(&node_info, &edge_info);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(node_info.size() == 2);

  std::shared_ptr<Tensor> nodes;
  s = graph.GetNodes(node_info[1].type, -1, &nodes);
  EXPECT_TRUE(s.IsOk());
  std::vector<NodeIdType> node_list;
  for (auto itr = nodes->begin<NodeIdType>(); itr != nodes->end<NodeIdType>(); ++itr) {
    node_list.push_back(*itr);
    if (node_list.size() >= 10) {
      break;
    }
  }
  std::shared_ptr<Tensor> neighbors;
  s = graph.GetAllNeighbors(node_list, node_info[0].type, &neighbors);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(neighbors->shape().ToString() == "<10,6>");
  TensorRow features;
  s = graph.GetNodeFeature(nodes, node_info[1].feature_type, &features);
  EXPECT_TRUE(s.IsOk());
  EXPECT_TRUE(features.size() == 3);
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
