/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_tensor.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"
#include <memory>
#include "ir/value.h"

namespace mindspore {
namespace parallel {
class TestEliminate : public UT::Common {
 public:
  void InitNode(std::shared_ptr<Graph> graph, int num_node);
  std::shared_ptr<Graph> MakeGraph(int num_node);
};

void TestEliminate::InitNode(std::shared_ptr<Graph> graph, int num_node) {
  Graph::NodeType NewNode;
  for (int i = 0; i < num_node; i++) {
    graph->nodes.push_back(NewNode);
    graph->nodes[i].name = "Add";
    graph->nodes[i].info = InfoType::kConstant;
  };
}

std::shared_ptr<Graph> TestEliminate::MakeGraph(int num_node) {
  // create graph
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  TestEliminate::InitNode(graph, num_node);

  // create edge
  std::vector<int64_t> edge_head = {1, 2, 2, 3};
  std::vector<int64_t> edge_tail = {0, 1, 3, 0};
  for (int i = 0; i < edge_head.size(); i++) {
    graph->nodes[edge_head[i]].node_out.push_back(edge_tail[i]);
    graph->nodes[edge_tail[i]].node_in.push_back(edge_head[i]);
  }

  // add node typr
  for (int i = 0; i < num_node; i++) {
    graph->nodes[i].apply.op_type = OperatorType::kRecAdd;
  }
  return graph;
}

/// Feature: test the eliminate aux during eliminate cost graph
/// Description: four add op in a cost graph
/// Expectation: successfully eliminate
TEST_F(TestEliminate, TestEliminateAux) {
  std::shared_ptr<Graph> graph = MakeGraph(4);
  std::shared_ptr<std::vector<std::vector<size_t>>> eli_list = std::make_shared<std::vector<std::vector<size_t>>>();
  for (size_t node_index = 0; node_index < graph->nodes.size(); node_index++) {
    auto type = graph->nodes[node_index].apply.op_type;
    if (EliminateOpType.find(type) != EliminateOpType.end()) {
      Eliminate_Aux(node_index, graph, eli_list);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore