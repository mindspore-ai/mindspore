/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/ops_info/matmul_info.h"

namespace mindspore {
namespace parallel {

using MatMulInfoPtr = std::shared_ptr<MatMulInfo>;

class TestCostGraph : public UT::Common {
 public:
  TestCostGraph() {
    matmul0 = nullptr;
    matmul1 = nullptr;
    matmul2 = nullptr;
    matmul3 = nullptr;
    matmul4 = nullptr;
    matmul5 = nullptr;
  }
  void SetUp();
  void TearDown() {}
  void ConstructDiamondGraph();
  void ConstructLinearGraph();
  void ConstructStarGraph();
  void ConstructSingleNodeGraph();
  void ConstructStarGraph2();

  MatMulInfoPtr matmul0;
  MatMulInfoPtr matmul1;
  MatMulInfoPtr matmul2;
  MatMulInfoPtr matmul3;
  MatMulInfoPtr matmul4;
  MatMulInfoPtr matmul5;
  CostGraph cost_graph;
};

void TestCostGraph::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 10; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(8);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  // matmul0
  ValuePtr transpose_a_0 = MakeValue(false);
  ValuePtr transpose_b_0 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_0 = {{"transpose_a", transpose_a_0}, {"transpose_b", transpose_b_0}};
  Shapes inputs_shape_0 = {{32, 16}, {16, 16}};
  Shapes outputs_shape_0 = {{32, 16}};
  matmul0 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_0, outputs_shape_0, attr_0);
  matmul0->set_outputs_type({kFloat32});

  // matmul1
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};
  Shapes inputs_shape_1 = {{8, 16}, {16, 32}};
  Shapes outputs_shape_1 = {{8, 32}};
  matmul1 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);
  matmul1->set_outputs_type({kFloat32});

  // matmul2
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpose_b_2}};
  Shapes inputs_shape_2 = {{8, 32}, {32, 16}};
  Shapes outputs_shape_2 = {{8, 16}};
  matmul2 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_2, outputs_shape_2, attr_2);
  matmul2->set_outputs_type({kFloat32});

  // matmul3
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpose_b_3 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_3}};
  Shapes inputs_shape_3 = {{16, 8}, {8, 32}};
  Shapes outputs_shape_3 = {{16, 32}};
  matmul3 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_3);
  matmul3->set_outputs_type({kFloat32});

  // matmul4
  ValuePtr transpose_a_4 = MakeValue(false);
  ValuePtr transpose_b_4 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_4 = {{"transpose_a", transpose_a_4}, {"transpose_b", transpose_b_4}};
  Shapes inputs_shape_4 = {{8, 16}, {16, 32}};
  Shapes outputs_shape_4 = {{8, 32}};
  matmul4 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_4, outputs_shape_4, attr_4);
  matmul4->set_outputs_type({kFloat32});

  // matmul5
  ValuePtr transpose_a_5 = MakeValue(false);
  ValuePtr transpose_b_5 = MakeValue(true);
  mindspore::HashMap<std::string, ValuePtr> attr_5 = {{"transpose_a", transpose_a_5}, {"transpose_b", transpose_b_5}};
  Shapes inputs_shape_5 = {{8, 32}, {8, 32}};
  Shapes outputs_shape_5 = {{8, 8}};
  matmul5 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_5, outputs_shape_5, attr_5);
  matmul5->set_outputs_type({kFloat32});
}

void TestCostGraph::ConstructStarGraph2() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m0_m2 = std::make_shared<Edge>(edge_name, matmul0, matmul2, 0, 1, false);
  std::shared_ptr<Edge> edge_m1_m3 = std::make_shared<Edge>(edge_name, matmul1, matmul3, 0, 1, false);

  matmul0->GenerateStrategies(0);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  cost_graph.AddOperator(matmul0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul2);
  cost_graph.AddOperator(matmul3);

  edge_m0_m2->InitEdgeCost();
  edge_m1_m2->InitEdgeCost();
  edge_m1_m3->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);
  matmul0->AddSuccEdge(edge_m0_m2);
  matmul2->AddPrevEdge(edge_m0_m2);
  cost_graph.AddEdge(matmul0, matmul2, edge_m0_m2);
  matmul1->AddSuccEdge(edge_m1_m3);
  matmul3->AddPrevEdge(edge_m1_m3);
  cost_graph.AddEdge(matmul1, matmul3, edge_m1_m3);
}

void TestCostGraph::ConstructDiamondGraph() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  std::shared_ptr<Edge> edge_m1_m3 = std::make_shared<Edge>(edge_name, matmul1, matmul3, 0, 1, false);
  std::shared_ptr<Edge> edge_m3_m4 = std::make_shared<Edge>(edge_name, matmul3, matmul4, 0, 1, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  edge_m1_m2->InitEdgeCost();
  edge_m2_m4->InitEdgeCost();
  edge_m1_m3->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  matmul1->AddSuccEdge(edge_m1_m3);
  matmul3->AddPrevEdge(edge_m1_m3);
  matmul3->AddSuccEdge(edge_m3_m4);
  matmul4->AddPrevEdge(edge_m3_m4);
}

void TestCostGraph::ConstructLinearGraph() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul2);
  cost_graph.AddOperator(matmul4);
  edge_m1_m2->InitEdgeCost();
  edge_m2_m4->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);
  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  cost_graph.AddEdge(matmul2, matmul4, edge_m2_m4);
}

void TestCostGraph::ConstructStarGraph() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  std::shared_ptr<Edge> edge_m3_m4 = std::make_shared<Edge>(edge_name, matmul3, matmul4, 0, 1, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul2);
  cost_graph.AddOperator(matmul3);
  cost_graph.AddOperator(matmul4);

  edge_m1_m2->InitEdgeCost();
  edge_m2_m4->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);
  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  cost_graph.AddEdge(matmul2, matmul4, edge_m2_m4);
  matmul3->AddSuccEdge(edge_m3_m4);
  matmul4->AddPrevEdge(edge_m3_m4);
  cost_graph.AddEdge(matmul3, matmul4, edge_m3_m4);
}

void TestCostGraph::ConstructSingleNodeGraph() {
  matmul1->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
}

TEST_F(TestCostGraph, DISABLED_test_CheckMergeElimination) {
  ConstructStarGraph();
  ASSERT_EQ(cost_graph.CheckMergeElimination().get(), matmul1.get());
  cost_graph.EliminationOp(matmul2);
  ASSERT_EQ(cost_graph.CheckMergeElimination().get(), matmul1.get());
  cost_graph.EliminationMerge(matmul1);
}

TEST_F(TestCostGraph, DISABLED_test_CheckContractAndMergeElimination) {
  ConstructStarGraph2();
  ASSERT_EQ(cost_graph.CheckMergeElimination().get(), matmul0.get());
  cost_graph.EliminationMerge(matmul0);
  ASSERT_EQ(cost_graph.CheckContractElimination().get(), matmul2.get());
}

TEST_F(TestCostGraph, DISABLED_test_EliminationMerge) {
  ConstructStarGraph();
  ASSERT_EQ(cost_graph.EliminationMerge(matmul3).get(), matmul4.get());
  ASSERT_EQ(matmul3->is_alive(), false);
}

TEST_F(TestCostGraph, DISABLED_test_SearchStrategy_for_single_node_graph) {
  ConstructSingleNodeGraph();
  cost_graph.SearchStrategy();
  auto cost = matmul1->selected_cost();
}

TEST_F(TestCostGraph, test_IsOperatorInCostGraph) {
  CostGraph entire_cost_graph;
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  entire_cost_graph.AddOperator(matmul1);
  entire_cost_graph.AddOperator(matmul2);
  entire_cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);

  ASSERT_EQ(entire_cost_graph.IsOperatorInCostGraph(matmul1), true);
  ASSERT_EQ(entire_cost_graph.IsOperatorInCostGraph(matmul2), true);
  ASSERT_EQ(entire_cost_graph.IsOperatorInCostGraph(matmul3), false);
  auto edges = entire_cost_graph.GetOriginalEdgeBetweenOperators(matmul1, matmul2);
  ASSERT_EQ(edges[0], edge_m1_m2);
}

TEST_F(TestCostGraph, test_ConstructConnectedComponents) {
  CostGraph entire_cost_graph;
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  std::shared_ptr<Edge> edge_m3_m4 = std::make_shared<Edge>(edge_name, matmul3, matmul4, 0, 0, false);
  matmul3->AddSuccEdge(edge_m3_m4);
  matmul4->AddPrevEdge(edge_m3_m4);
  entire_cost_graph.AddOperator(matmul1);
  entire_cost_graph.AddOperator(matmul2);
  entire_cost_graph.AddOperator(matmul3);
  entire_cost_graph.AddOperator(matmul4);
  entire_cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);
  entire_cost_graph.AddEdge(matmul3, matmul4, edge_m3_m4);
  // test for successfully construct connected components
  std::vector<std::shared_ptr<CostGraph>> connected_coms =
    entire_cost_graph.ConstructConnectedComponents(entire_cost_graph.GetOperators());
  ASSERT_EQ(connected_coms.size(), 2);
  ASSERT_EQ(connected_coms[0]->GetOperators().size(), 2);
  ASSERT_EQ(connected_coms[1]->GetOperators().size(), 2);
  ASSERT_EQ(connected_coms[0]->GetOriginalEdgeBetweenOperators(matmul1, matmul2)[0].get(), edge_m1_m2.get());
  ASSERT_EQ(connected_coms[1]->GetOriginalEdgeBetweenOperators(matmul3, matmul4)[0].get(), edge_m3_m4.get());
}

TEST_F(TestCostGraph, test_SelectCostListWithMinTrainingTimeMultiple) {
  CostGraph entire_cost_graph;
  double memory = 1024.0;
  CostPtrList clist_1, clist_2;
  std::vector<CostPtrList> all_list;

  auto cost1_1 = std::make_shared<Cost>(10, 20);
  cost1_1->communication_without_parameter_ = 20;
  clist_1.push_back(cost1_1);
  auto cost1_2 = std::make_shared<Cost>(100, 10);
  cost1_2->communication_without_parameter_ = 0.0;
  clist_1.push_back(cost1_2);
  all_list.push_back(clist_1);

  auto cost2_1 = std::make_shared<Cost>(1010, 20);
  cost2_1->communication_without_parameter_ = 10;
  clist_2.push_back(cost2_1);
  auto cost2_2 = std::make_shared<Cost>(1050, 20);
  cost2_2->communication_without_parameter_ = 10;
  clist_2.push_back(cost2_2);
  all_list.push_back(clist_2);

  auto ret_list = entire_cost_graph.SelectCostListWithMinTrainingTimeMultiple(all_list, memory);
  ASSERT_EQ(ret_list.size(), 2);
  ASSERT_DOUBLE_EQ(ret_list[0]->computation_cost_, 10);
  ASSERT_DOUBLE_EQ(ret_list[1]->computation_cost_, 1010);
}

TEST_F(TestCostGraph, DISABLED_test_CheckOpElimination) {
  ConstructLinearGraph();
  ASSERT_EQ(cost_graph.CheckOpElimination().get(), matmul2.get());
}

TEST_F(TestCostGraph, DISABLED_test_CheckEdgesElimination) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m5 = std::make_shared<Edge>(edge_name, matmul1, matmul5, 0, 0, false);
  std::shared_ptr<Edge> edge_m1_m5_2 = std::make_shared<Edge>(edge_name, matmul1, matmul5, 0, 1, false);
  matmul1->GenerateStrategies(0);
  matmul5->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul5);

  matmul1->AddSuccEdge(edge_m1_m5);
  matmul1->AddSuccEdge(edge_m1_m5_2);
  matmul5->AddPrevEdge(edge_m1_m5);
  matmul5->AddPrevEdge(edge_m1_m5_2);
  cost_graph.AddEdge(matmul1, matmul5, edge_m1_m5);
  cost_graph.AddEdge(matmul1, matmul5, edge_m1_m5_2);
  ASSERT_EQ(cost_graph.CheckEdgeElimination().size(), 2);
  ASSERT_EQ(cost_graph.CheckEdgeElimination()[0].get(), edge_m1_m5.get());
  ASSERT_EQ(cost_graph.CheckEdgeElimination()[1].get(), edge_m1_m5_2.get());
}

TEST_F(TestCostGraph, DISABLED_test_CreateFinalCostList_AND_Select) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul2);

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);

  ASSERT_EQ(edge_m1_m2->InitEdgeCost(), SUCCESS);
  cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);
  auto cost_list = cost_graph.CreateFinalCostList(matmul1, edge_m1_m2, matmul2);
  const auto device_mem_capacity = CostModelContext::GetInstance()->device_memory_capacity();
  cost_graph.SelectCostWithMinInferenceTime(cost_list, device_mem_capacity);
}

TEST_F(TestCostGraph, DISABLED_test_EliminationOp) {
  ConstructLinearGraph();
  auto new_edge = cost_graph.EliminationOp(matmul2);
  ASSERT_EQ(new_edge.get(), matmul1->succ_edges()[0].get());
  ASSERT_EQ(new_edge.get(), matmul4->prev_edges()[0].get());
}

TEST_F(TestCostGraph, DISABLED_test_EliminationEdges) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m5 = std::make_shared<Edge>(edge_name, matmul1, matmul5, 0, 0, false);
  std::shared_ptr<Edge> edge_m1_m5_2 = std::make_shared<Edge>(edge_name, matmul1, matmul5, 0, 1, false);
  matmul1->GenerateStrategies(0);
  matmul5->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul5);
  edge_m1_m5->InitEdgeCost();
  edge_m1_m5_2->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m5);
  matmul1->AddSuccEdge(edge_m1_m5_2);
  matmul5->AddPrevEdge(edge_m1_m5);
  matmul5->AddPrevEdge(edge_m1_m5_2);
  cost_graph.AddEdge(matmul1, matmul5, edge_m1_m5);
  cost_graph.AddEdge(matmul1, matmul5, edge_m1_m5_2);

  std::vector<std::shared_ptr<Edge>> edges;
  edges.push_back(edge_m1_m5);
  edges.push_back(edge_m1_m5_2);

  auto new_edge = cost_graph.EliminationEdges(edges);
  ASSERT_EQ(new_edge.get(), matmul1->succ_edges()[0].get());
  ASSERT_EQ(new_edge.get(), matmul5->prev_edges()[0].get());
}

TEST_F(TestCostGraph, DISABLED_test_SearchStrategy) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul2);

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);

  ASSERT_EQ(edge_m1_m2->InitEdgeCost(), SUCCESS);
  cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);
  cost_graph.SearchStrategy();
}

TEST_F(TestCostGraph, DISABLED_test_SearchStrategyV2) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  cost_graph.AddOperator(matmul1);
  cost_graph.AddOperator(matmul2);

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);

  ASSERT_EQ(edge_m1_m2->InitEdgeCost(), SUCCESS);
  cost_graph.AddEdge(matmul1, matmul2, edge_m1_m2);

  cost_graph.EliminationMerge(matmul1);
  cost_graph.SearchStrategy();

  auto decision = matmul2->selected_cost()->decision_ptr_->cast<MergeEliminationDecisionPtr>();

  matmul1->SetSelectedStrategyAndCost(decision->merged_op_strategy_, decision->merged_op_cost_);
  edge_m1_m2->set_selected_cost(decision->edge_cost_);
}
}  // namespace parallel
}  // namespace mindspore
