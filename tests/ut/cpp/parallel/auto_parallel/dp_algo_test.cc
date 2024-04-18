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
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/ops_info/tmp_identity_info.h"
#include "frontend/parallel/auto_parallel/dp_algo_costmodel.h"

namespace mindspore {
namespace parallel {

using MatMulInfoPtr = std::shared_ptr<MatMulInfo>;
using ActivationPtr = std::shared_ptr<ActivationInfo>;
using TmpIdentityPtr = std::shared_ptr<TmpIdentityInfo>;

class TestDPAlgo : public UT::Common {
 public:
  TestDPAlgo() {
    matmul0 = nullptr;
    matmul1 = nullptr;
    matmul2 = nullptr;
    matmul3 = nullptr;
    matmul4 = nullptr;
    mm1_ptr = nullptr;
    mm2_ptr = nullptr;
    mm3_ptr = nullptr;
    mm4_ptr = nullptr;
    mm5_ptr = nullptr;
    mm6_ptr = nullptr;
    mm7_ptr = nullptr;
    relu1_ptr = nullptr;
    relu2_ptr = nullptr;
    relu3_ptr = nullptr;
    relu4_ptr = nullptr;
    relu5_ptr = nullptr;
    matmul7 = nullptr;
    matmul8 = nullptr;

    tmp_identity_ptr = nullptr;
    tmp_identity_ptr1 = nullptr;
    tmp_identity_ptr2 = nullptr;
    edge_m1_m2 = nullptr;
    edge_m1_m3 = nullptr;
    edge_m1_m4 = nullptr;
    edge_m2_r1 = nullptr;
    edge_m3_r2 = nullptr;
    edge_m4_r3 = nullptr;
    edge_r1_m5 = nullptr;
    edge_r2_m5 = nullptr;
    edge_m5_r4 = nullptr;
    edge_r4_m6 = nullptr;
    edge_r3_m6 = nullptr;
    edge_m6_r5 = nullptr;

    edge_m2_m3 = nullptr;
    edge_i1_m1 = nullptr;
    edge_i1_m2 = nullptr;
    edge_m3_m1 = nullptr;
    edge_i2_m4 = nullptr;
    edge_i2_m5 = nullptr;
    edge_m4_m5 = nullptr;
    edge_m3_m4 = nullptr;
    edge_i0_m3 = nullptr;
    edge_i0_m6 = nullptr;
    edge_i1_m4 = nullptr;
    edge_i2_m7 = nullptr;
    edge_m1_m5 = nullptr;
    edge_m5_m6 = nullptr;
    edge_m6_m7 = nullptr;
    edge_m7_m8 = nullptr;
    cost_graph = nullptr;
  }
  void SetUp();
  void TearDown() {}
  void ConstructDiamondGraph();
  void ConstructMMRGraph();
  void ConstructIdentityDiamondGraph();
  void ConstructStarGraph();
  void ConstructDoubleStarGraph();
  void ConstructTwoSeparateGraphs();
  void ConstructTwoSeparateSingleNodeGraph();
  void ConstructThreeSeparateGraphs();
  void ConstructStarGraph2();
  void ConstructStarGraph3();
  void ConstructTwoSeparateGraphs2();
  void ConstructTriangleGraph();
  void ConstructTriangleGraph2();
  void ConstructBatmanGraph();
  void ConstructTwoLargeMatMul();

  MatMulInfoPtr matmul0;
  MatMulInfoPtr matmul1;
  MatMulInfoPtr matmul2;
  MatMulInfoPtr matmul3;
  MatMulInfoPtr matmul4;
  MatMulInfoPtr matmul5;
  MatMulInfoPtr matmul6;
  MatMulInfoPtr matmul7;
  MatMulInfoPtr matmul8;
  MatMulInfoPtr mm1_ptr, mm2_ptr, mm3_ptr, mm4_ptr, mm5_ptr, mm6_ptr, mm7_ptr;
  ActivationPtr relu1_ptr, relu2_ptr, relu3_ptr, relu4_ptr, relu5_ptr;
  TmpIdentityPtr tmp_identity_ptr;
  TmpIdentityPtr tmp_identity_ptr1;
  TmpIdentityPtr tmp_identity_ptr2;
  std::shared_ptr<Edge> edge_m1_m2;
  std::shared_ptr<Edge> edge_m1_m3;
  std::shared_ptr<Edge> edge_m1_m4;
  std::shared_ptr<Edge> edge_m2_r1;
  std::shared_ptr<Edge> edge_m3_r2;
  std::shared_ptr<Edge> edge_m4_r3;
  std::shared_ptr<Edge> edge_r1_m5;
  std::shared_ptr<Edge> edge_r2_m5;
  std::shared_ptr<Edge> edge_m5_r4;
  std::shared_ptr<Edge> edge_r4_m6;
  std::shared_ptr<Edge> edge_r3_m6;
  std::shared_ptr<Edge> edge_m6_r5;
  std::shared_ptr<Edge> edge_i2_m4;
  std::shared_ptr<Edge> edge_i2_m5;
  std::shared_ptr<Edge> edge_m4_m5;
  std::shared_ptr<Edge> edge_m3_m4;

  std::shared_ptr<Edge> edge_m2_m3;
  std::shared_ptr<Edge> edge_i1_m1;
  std::shared_ptr<Edge> edge_i1_m2;
  std::shared_ptr<Edge> edge_m3_m1;

  std::shared_ptr<Edge> edge_i0_m3;
  std::shared_ptr<Edge> edge_i0_m6;
  std::shared_ptr<Edge> edge_i1_m4;
  std::shared_ptr<Edge> edge_i2_m7;
  std::shared_ptr<Edge> edge_m1_m5;
  std::shared_ptr<Edge> edge_m5_m6;
  std::shared_ptr<Edge> edge_m6_m7;
  std::shared_ptr<Edge> edge_m7_m8;
  CostGraphPtr cost_graph;
};

void TestDPAlgo::SetUp() {
  cost_graph = std::make_shared<CostGraph>();
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
  Shapes inputs_shape_0 = {{128, 1024}, {1024, 4096}};
  Shapes outputs_shape_0 = {{4096, 1024}};
  matmul0 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_0, outputs_shape_0, attr_0);
  matmul0->set_name("MatMul0");
  matmul0->set_outputs_type({kFloat32});

  // matmul1
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};
  Shapes inputs_shape_1 = {{128, 1024}, {1024, 4096}};
  Shapes outputs_shape_1 = {{128, 4096}};
  matmul1 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);
  matmul1->set_name("MatMul1");
  matmul1->set_outputs_type({kFloat32});

  // matmul2
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpose_b_2}};
  Shapes inputs_shape_2 = {{128, 4096}, {4096, 1024}};
  Shapes outputs_shape_2 = {{128, 1024}};
  matmul2 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_2, outputs_shape_2, attr_2);
  matmul2->set_name("MatMul2");
  matmul2->set_outputs_type({kFloat32});

  // matmul3
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpose_b_3 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_3}};
  Shapes inputs_shape_3 = {{1024, 128}, {128, 4096}};
  Shapes outputs_shape_3 = {{1024, 4096}};
  matmul3 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_3);
  matmul3->set_name("MatMul3");
  matmul3->set_outputs_type({kFloat32});

  // matmul4
  ValuePtr transpose_a_4 = MakeValue(false);
  ValuePtr transpose_b_4 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_4 = {{"transpose_a", transpose_a_4}, {"transpose_b", transpose_b_4}};
  Shapes inputs_shape_4 = {{128, 1024}, {1024, 4096}};
  Shapes outputs_shape_4 = {{128, 4096}};
  matmul4 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_4, outputs_shape_4, attr_4);
  matmul4->set_name("MatMul4");
  matmul4->set_outputs_type({kFloat32});

  // matmul5
  ValuePtr transpose_a_5 = MakeValue(false);
  ValuePtr transpose_b_5 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_5 = {{"transpose_a", transpose_a_5}, {"transpose_b", transpose_b_5}};
  Shapes inputs_shape_5 = {{128, 4096}, {4096, 4096}};
  Shapes outputs_shape_5 = {{128, 4096}};
  matmul5 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_5, outputs_shape_5, attr_5);
  matmul5->set_name("MatMul5");
  matmul5->set_outputs_type({kFloat32});

  // matmul6
  ValuePtr transpose_a_6 = MakeValue(false);
  ValuePtr transpose_b_6 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_6 = {{"transpose_a", transpose_a_6}, {"transpose_b", transpose_b_6}};
  Shapes inputs_shape_6 = {{4096, 128}, {128, 1024}};
  Shapes outputs_shape_6 = {{4096, 1024}};
  matmul6 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_6, outputs_shape_6, attr_6);
  matmul6->set_name("MatMul6");
  matmul6->set_outputs_type({kFloat32});

  // matmul7
  ValuePtr transpose_a_7 = MakeValue(false);
  ValuePtr transpose_b_7 = MakeValue(true);
  mindspore::HashMap<std::string, ValuePtr> attr_7 = {{"transpose_a", transpose_a_7}, {"transpose_b", transpose_b_7}};
  Shapes inputs_shape_7 = {{64, 128}, {4096, 128}};
  Shapes outputs_shape_7 = {{64, 4096}};
  matmul7 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_7, outputs_shape_7, attr_7);
  matmul7->set_name("MatMul7");
  matmul7->set_outputs_type({kFloat32});

  // matmul8
  ValuePtr transpose_a_8 = MakeValue(false);
  ValuePtr transpose_b_8 = MakeValue(true);
  mindspore::HashMap<std::string, ValuePtr> attr_8 = {{"transpose_a", transpose_a_8}, {"transpose_b", transpose_b_8}};
  Shapes inputs_shape_8 = {{64, 4096}, {40960, 4096}};
  Shapes outputs_shape_8 = {{64, 40960}};
  matmul8 = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_8, outputs_shape_8, attr_8);
  matmul8->set_name("MatMul8");
  matmul8->set_outputs_type({kFloat32});
}

void TestDPAlgo::ConstructTwoLargeMatMul() {
  std::string edge_matmul_matmul_name = "MatMul-MatMul";
  edge_m7_m8 = std::make_shared<Edge>(edge_matmul_matmul_name, matmul7, matmul8, 0, 0, false);

  matmul7->GenerateStrategies(0);
  matmul8->GenerateStrategies(0);

  cost_graph->AddOperator(matmul7);
  cost_graph->AddOperator(matmul8);

  edge_m7_m8->InitEdgeCost();

  matmul7->AddSuccEdge(edge_m7_m8);
  matmul8->AddPrevEdge(edge_m7_m8);
  cost_graph->AddEdge(matmul7, matmul8, edge_m7_m8);
}

void TestDPAlgo::ConstructBatmanGraph() {
  std::string edge_matmul_matmul_name = "MatMul-MatMul";
  std::string edge_iden_matmul_name = "TmpIdentity-MatMul";

  mindspore::HashMap<std::string, ValuePtr> attr = {};
  Shapes inputs_shape = {{64, 64}};
  Shapes outputs_shape = {{64, 64}};
  tmp_identity_ptr1 = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
  tmp_identity_ptr1->set_name("identity_info1");
  tmp_identity_ptr1->set_outputs_type({kFloat32});

  tmp_identity_ptr2 = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
  tmp_identity_ptr2->set_name("identity_info2");
  tmp_identity_ptr2->set_outputs_type({kFloat32});

  tmp_identity_ptr = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
  tmp_identity_ptr->set_name("identity_info");
  tmp_identity_ptr->set_outputs_type({kFloat32});

  // mm1_ptr
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};
  Shapes inputs_shape_1 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_1 = {{64, 64}};
  mm1_ptr = std::make_shared<MatMulInfo>("matmul_info1", inputs_shape_1, outputs_shape_1, attr_1);
  mm1_ptr->set_outputs_type({kFloat32});

  // mm2_ptr
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpose_b_2}};
  Shapes inputs_shape_2 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_2 = {{64, 64}};
  mm2_ptr = std::make_shared<MatMulInfo>("matmul_info2", inputs_shape_2, outputs_shape_2, attr_2);
  mm2_ptr->set_outputs_type({kFloat32});

  // mm3_ptr
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpose_b_3 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_3}};
  Shapes inputs_shape_3 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_3 = {{64, 64}};
  mm3_ptr = std::make_shared<MatMulInfo>("matmul_info3", inputs_shape_3, outputs_shape_3, attr_3);
  mm3_ptr->set_outputs_type({kFloat32});

  // mm4_ptr
  ValuePtr transpose_a_4 = MakeValue(false);
  ValuePtr transpose_b_4 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_4 = {{"transpose_a", transpose_a_4}, {"transpose_b", transpose_b_4}};
  Shapes inputs_shape_4 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_4 = {{64, 64}};
  mm4_ptr = std::make_shared<MatMulInfo>("matmul_info4", inputs_shape_4, outputs_shape_4, attr_4);
  mm4_ptr->set_outputs_type({kFloat32});

  // mm5_ptr
  ValuePtr transpose_a_5 = MakeValue(false);
  ValuePtr transpose_b_5 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_5 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_5}};
  Shapes inputs_shape_5 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_5 = {{64, 64}};
  mm5_ptr = std::make_shared<MatMulInfo>("matmul_info5", inputs_shape_5, outputs_shape_5, attr_5);
  mm5_ptr->set_outputs_type({kFloat32});

  // mm6_ptr
  ValuePtr transpose_a_6 = MakeValue(false);
  ValuePtr transpose_b_6 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_6 = {{"transpose_a", transpose_a_6}, {"transpose_b", transpose_b_6}};
  Shapes inputs_shape_6 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_6 = {{64, 64}};
  mm6_ptr = std::make_shared<MatMulInfo>("matmul_info6", inputs_shape_6, outputs_shape_6, attr_6);
  mm6_ptr->set_outputs_type({kFloat32});

  // mm7_ptr
  ValuePtr transpose_a_7 = MakeValue(false);
  ValuePtr transpose_b_7 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_7 = {{"transpose_a", transpose_a_7}, {"transpose_b", transpose_a_7}};
  Shapes inputs_shape_7 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_7 = {{64, 64}};
  mm7_ptr = std::make_shared<MatMulInfo>("matmul_info7", inputs_shape_7, outputs_shape_7, attr_7);
  mm7_ptr->set_outputs_type({kFloat32});

  // create edges
  edge_i0_m3 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr, mm3_ptr, 0, 0, false, true);
  edge_i0_m6 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr, mm6_ptr, 0, 0, false, true);
  edge_i1_m2 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr1, mm2_ptr, 0, 0, false, true);
  edge_i1_m4 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr1, mm4_ptr, 0, 0, false, true);
  edge_i2_m5 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr2, mm5_ptr, 0, 0, false, true);
  edge_i2_m7 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr2, mm7_ptr, 0, 0, false, true);

  edge_m1_m2 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm2_ptr, 0, 1, false, false);
  edge_m1_m5 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm5_ptr, 0, 1, false, false);
  edge_m2_m3 = std::make_shared<Edge>(edge_matmul_matmul_name, mm2_ptr, mm3_ptr, 0, 1, false, false);
  edge_m3_m4 = std::make_shared<Edge>(edge_matmul_matmul_name, mm3_ptr, mm4_ptr, 0, 1, false, false);
  edge_m5_m6 = std::make_shared<Edge>(edge_matmul_matmul_name, mm5_ptr, mm6_ptr, 0, 1, false, false);
  edge_m6_m7 = std::make_shared<Edge>(edge_matmul_matmul_name, mm6_ptr, mm7_ptr, 0, 1, false, false);

  // init operators
  tmp_identity_ptr->GenerateStrategies(0);
  tmp_identity_ptr1->GenerateStrategies(0);
  tmp_identity_ptr2->GenerateStrategies(0);
  mm1_ptr->GenerateStrategies(0);
  mm2_ptr->GenerateStrategies(0);
  mm3_ptr->GenerateStrategies(0);
  mm4_ptr->GenerateStrategies(0);
  mm5_ptr->GenerateStrategies(0);
  mm6_ptr->GenerateStrategies(0);
  mm7_ptr->GenerateStrategies(0);

  // add operators to costgraph
  cost_graph->AddOperator(tmp_identity_ptr);
  cost_graph->AddOperator(tmp_identity_ptr1);
  cost_graph->AddOperator(tmp_identity_ptr2);
  cost_graph->AddOperator(mm1_ptr);
  cost_graph->AddOperator(mm2_ptr);
  cost_graph->AddOperator(mm3_ptr);
  cost_graph->AddOperator(mm4_ptr);
  cost_graph->AddOperator(mm5_ptr);
  cost_graph->AddOperator(mm6_ptr);
  cost_graph->AddOperator(mm7_ptr);

  // init edge cost
  edge_i0_m3->InitEdgeCost();
  edge_i0_m6->InitEdgeCost();
  edge_i1_m2->InitEdgeCost();
  edge_i1_m4->InitEdgeCost();
  edge_i2_m5->InitEdgeCost();
  edge_i2_m7->InitEdgeCost();

  edge_m1_m2->InitEdgeCost();
  edge_m1_m5->InitEdgeCost();
  edge_m2_m3->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();
  edge_m5_m6->InitEdgeCost();
  edge_m6_m7->InitEdgeCost();

  // add edges to costgraph
  tmp_identity_ptr->AddSuccEdge(edge_i0_m3);
  mm3_ptr->AddPrevEdge(edge_i0_m3);
  cost_graph->AddEdge(tmp_identity_ptr, mm3_ptr, edge_i0_m3);

  tmp_identity_ptr->AddSuccEdge(edge_i0_m6);
  mm6_ptr->AddPrevEdge(edge_i0_m6);
  cost_graph->AddEdge(tmp_identity_ptr, mm6_ptr, edge_i0_m6);

  tmp_identity_ptr1->AddSuccEdge(edge_i1_m2);
  mm2_ptr->AddPrevEdge(edge_i1_m2);
  cost_graph->AddEdge(tmp_identity_ptr1, mm2_ptr, edge_i1_m2);

  tmp_identity_ptr1->AddSuccEdge(edge_i1_m4);
  mm4_ptr->AddPrevEdge(edge_i1_m4);
  cost_graph->AddEdge(tmp_identity_ptr1, mm4_ptr, edge_i1_m4);

  tmp_identity_ptr2->AddSuccEdge(edge_i2_m5);
  mm5_ptr->AddPrevEdge(edge_i2_m5);
  cost_graph->AddEdge(tmp_identity_ptr2, mm5_ptr, edge_i2_m5);

  tmp_identity_ptr2->AddSuccEdge(edge_i2_m7);
  mm7_ptr->AddPrevEdge(edge_i2_m7);
  cost_graph->AddEdge(tmp_identity_ptr2, mm7_ptr, edge_i2_m7);

  mm1_ptr->AddSuccEdge(edge_m1_m2);
  mm2_ptr->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(mm1_ptr, mm2_ptr, edge_m1_m2);

  mm1_ptr->AddSuccEdge(edge_m1_m5);
  mm5_ptr->AddPrevEdge(edge_m1_m5);
  cost_graph->AddEdge(mm1_ptr, mm5_ptr, edge_m1_m5);

  mm2_ptr->AddSuccEdge(edge_m2_m3);
  mm3_ptr->AddPrevEdge(edge_m2_m3);
  cost_graph->AddEdge(mm2_ptr, mm3_ptr, edge_m2_m3);

  mm3_ptr->AddSuccEdge(edge_m3_m4);
  mm4_ptr->AddPrevEdge(edge_m3_m4);
  cost_graph->AddEdge(mm3_ptr, mm4_ptr, edge_m3_m4);

  mm5_ptr->AddSuccEdge(edge_m5_m6);
  mm6_ptr->AddPrevEdge(edge_m5_m6);
  cost_graph->AddEdge(mm5_ptr, mm6_ptr, edge_m5_m6);

  mm6_ptr->AddSuccEdge(edge_m6_m7);
  mm7_ptr->AddPrevEdge(edge_m6_m7);
  cost_graph->AddEdge(mm6_ptr, mm7_ptr, edge_m6_m7);
}

void TestDPAlgo::ConstructTriangleGraph() {
  mindspore::HashMap<std::string, ValuePtr> attr = {};
  Shapes inputs_shape = {{64, 64}};
  Shapes outputs_shape = {{64, 64}};
  tmp_identity_ptr1 = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
  tmp_identity_ptr1->set_name("identity_info1");
  tmp_identity_ptr1->set_outputs_type({kFloat32});

  // mm6_ptr
  ValuePtr transpose_a_6 = MakeValue(false);
  ValuePtr transpose_b_6 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_6 = {{"transpose_a", transpose_a_6}, {"transpose_b", transpose_b_6}};
  Shapes inputs_shape_6 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_6 = {{64, 64}};
  mm6_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_6, outputs_shape_6, attr_6);
  mm6_ptr->set_outputs_type({kFloat32});

  tmp_identity_ptr2 = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
  tmp_identity_ptr2->set_name("identity_info2");
  tmp_identity_ptr2->set_outputs_type({kFloat32});

  // mm1_ptr
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};
  Shapes inputs_shape_1 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_1 = {{64, 64}};
  mm1_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);
  mm1_ptr->set_outputs_type({kFloat32});

  // mm2_ptr
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpose_b_2}};
  Shapes inputs_shape_2 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_2 = {{64, 64}};
  mm2_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_2, outputs_shape_2, attr_2);
  mm2_ptr->set_outputs_type({kFloat32});

  // mm3_ptr
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpose_b_3 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_3}};
  Shapes inputs_shape_3 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_3 = {{64, 64}};
  mm3_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_3);
  mm3_ptr->set_outputs_type({kFloat32});

  // mm4_ptr
  ValuePtr transpose_a_4 = MakeValue(false);
  ValuePtr transpose_b_4 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_4 = {{"transpose_a", transpose_a_4}, {"transpose_b", transpose_b_4}};
  Shapes inputs_shape_4 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_4 = {{64, 64}};
  mm4_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_4, outputs_shape_4, attr_4);
  mm4_ptr->set_outputs_type({kFloat32});

  // mm5_ptr
  ValuePtr transpose_a_5 = MakeValue(false);
  ValuePtr transpose_b_5 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_5 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_5}};
  Shapes inputs_shape_5 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_5 = {{64, 64}};
  mm5_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_5, outputs_shape_5, attr_5);
  mm5_ptr->set_outputs_type({kFloat32});

  // create edges
  std::string edge_matmul_matmul_name = "MatMul-MatMul";
  std::string edge_iden_matmul_name = "TmpIdentity-MatMul";
  edge_i1_m1 = std::make_shared<Edge>(edge_matmul_matmul_name, mm6_ptr, mm1_ptr, 0, 0, false, false);
  edge_i1_m2 = std::make_shared<Edge>(edge_matmul_matmul_name, mm6_ptr, mm2_ptr, 0, 0, false, false);
  edge_m1_m2 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm2_ptr, 0, 1, false, false);
  edge_m3_m1 = std::make_shared<Edge>(edge_matmul_matmul_name, mm3_ptr, mm1_ptr, 0, 1, false, false);
  edge_i2_m4 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr2, mm4_ptr, 0, 0, false, true);
  edge_i2_m5 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr2, mm5_ptr, 0, 0, false, true);
  edge_m4_m5 = std::make_shared<Edge>(edge_matmul_matmul_name, mm4_ptr, mm5_ptr, 0, 1, false, false);
  edge_m3_m4 = std::make_shared<Edge>(edge_matmul_matmul_name, mm3_ptr, mm4_ptr, 0, 1, false, false);

  // init operators
  mm6_ptr->GenerateStrategies(0);
  mm1_ptr->GenerateStrategies(0);
  mm2_ptr->GenerateStrategies(0);
  mm3_ptr->GenerateStrategies(0);
  tmp_identity_ptr2->GenerateStrategies(0);
  mm4_ptr->GenerateStrategies(0);
  mm5_ptr->GenerateStrategies(0);

  // add operators to costgraph
  cost_graph->AddOperator(mm6_ptr);
  cost_graph->AddOperator(mm1_ptr);
  cost_graph->AddOperator(mm2_ptr);
  cost_graph->AddOperator(mm3_ptr);
  cost_graph->AddOperator(tmp_identity_ptr2);
  cost_graph->AddOperator(mm4_ptr);
  cost_graph->AddOperator(mm5_ptr);

  // init edge cost
  edge_i1_m1->InitEdgeCost();
  edge_i1_m2->InitEdgeCost();
  edge_m1_m2->InitEdgeCost();
  edge_m3_m1->InitEdgeCost();

  edge_i2_m4->InitEdgeCost();
  edge_i2_m5->InitEdgeCost();
  edge_m4_m5->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();

  // add edges to costgraph
  mm6_ptr->AddSuccEdge(edge_i1_m1);
  mm1_ptr->AddPrevEdge(edge_i1_m1);
  cost_graph->AddEdge(mm6_ptr, mm1_ptr, edge_i1_m1);

  mm6_ptr->AddSuccEdge(edge_i1_m2);
  mm2_ptr->AddPrevEdge(edge_i1_m2);
  cost_graph->AddEdge(mm6_ptr, mm2_ptr, edge_i1_m2);

  mm1_ptr->AddSuccEdge(edge_m1_m2);
  mm2_ptr->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(mm1_ptr, mm2_ptr, edge_m1_m2);

  mm3_ptr->AddSuccEdge(edge_m3_m1);
  mm1_ptr->AddPrevEdge(edge_m3_m1);
  cost_graph->AddEdge(mm3_ptr, mm1_ptr, edge_m3_m1);

  tmp_identity_ptr2->AddSuccEdge(edge_i2_m4);
  mm4_ptr->AddPrevEdge(edge_i2_m4);
  cost_graph->AddEdge(tmp_identity_ptr2, mm4_ptr, edge_i2_m4);

  tmp_identity_ptr2->AddSuccEdge(edge_i2_m5);
  mm5_ptr->AddPrevEdge(edge_i2_m5);
  cost_graph->AddEdge(tmp_identity_ptr2, mm5_ptr, edge_i2_m5);

  mm4_ptr->AddSuccEdge(edge_m4_m5);
  mm5_ptr->AddPrevEdge(edge_m4_m5);
  cost_graph->AddEdge(mm4_ptr, mm5_ptr, edge_m4_m5);

  mm3_ptr->AddSuccEdge(edge_m3_m4);
  mm4_ptr->AddPrevEdge(edge_m3_m4);
  cost_graph->AddEdge(mm3_ptr, mm4_ptr, edge_m3_m4);
}

void TestDPAlgo::ConstructTriangleGraph2() {
  mindspore::HashMap<std::string, ValuePtr> attr = {};
  Shapes inputs_shape = {{64, 64}};
  Shapes outputs_shape = {{64, 64}};
  tmp_identity_ptr1 = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
  tmp_identity_ptr1->set_name("identity_info1");
  tmp_identity_ptr1->set_outputs_type({kFloat32});

  // mm1_ptr
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};
  Shapes inputs_shape_1 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_1 = {{64, 64}};
  mm1_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);
  mm1_ptr->set_outputs_type({kFloat32});

  // mm2_ptr
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpose_b_2}};
  Shapes inputs_shape_2 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_2 = {{64, 64}};
  mm2_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_2, outputs_shape_2, attr_2);
  mm2_ptr->set_outputs_type({kFloat32});

  // mm3_ptr
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpose_b_3 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_3}};
  Shapes inputs_shape_3 = {{64, 64}, {64, 64}};
  Shapes outputs_shape_3 = {{64, 64}};
  mm3_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_3);
  mm3_ptr->set_outputs_type({kFloat32});

  // create edges
  std::string edge_matmul_matmul_name = "MatMul-MatMul";
  std::string edge_iden_matmul_name = "TmpIdentity-MatMul";
  edge_i1_m1 = std::make_shared<Edge>(edge_matmul_matmul_name, tmp_identity_ptr1, mm1_ptr, 0, 0, false, true);
  edge_i1_m2 = std::make_shared<Edge>(edge_matmul_matmul_name, tmp_identity_ptr1, mm2_ptr, 0, 0, false, true);
  edge_m1_m2 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm2_ptr, 0, 1, false, false);
  edge_m3_m1 = std::make_shared<Edge>(edge_matmul_matmul_name, mm3_ptr, mm1_ptr, 0, 1, false, false);

  // init operators
  tmp_identity_ptr1->GenerateStrategies(0);
  mm1_ptr->GenerateStrategies(0);
  mm2_ptr->GenerateStrategies(0);
  mm3_ptr->GenerateStrategies(0);

  // add operators to costgraph
  cost_graph->AddOperator(tmp_identity_ptr1);
  cost_graph->AddOperator(mm1_ptr);
  cost_graph->AddOperator(mm2_ptr);
  cost_graph->AddOperator(mm3_ptr);

  // init edge cost
  edge_i1_m1->InitEdgeCost();
  edge_i1_m2->InitEdgeCost();
  edge_m1_m2->InitEdgeCost();
  edge_m3_m1->InitEdgeCost();

  // add edges to costgraph
  tmp_identity_ptr1->AddSuccEdge(edge_i1_m1);
  mm1_ptr->AddPrevEdge(edge_i1_m1);
  cost_graph->AddEdge(tmp_identity_ptr1, mm1_ptr, edge_i1_m1);

  tmp_identity_ptr1->AddSuccEdge(edge_i1_m2);
  mm2_ptr->AddPrevEdge(edge_i1_m2);
  cost_graph->AddEdge(tmp_identity_ptr1, mm2_ptr, edge_i1_m2);

  mm1_ptr->AddSuccEdge(edge_m1_m2);
  mm2_ptr->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(mm1_ptr, mm2_ptr, edge_m1_m2);

  mm3_ptr->AddSuccEdge(edge_m3_m1);
  mm1_ptr->AddPrevEdge(edge_m3_m1);
  cost_graph->AddEdge(mm3_ptr, mm1_ptr, edge_m3_m1);
}

void TestDPAlgo::ConstructStarGraph() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  std::shared_ptr<Edge> edge_m3_m4 = std::make_shared<Edge>(edge_name, matmul3, matmul4, 0, 1, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul3);
  cost_graph->AddOperator(matmul4);

  edge_m1_m2->InitEdgeCost();
  edge_m2_m4->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(matmul1, matmul2, edge_m1_m2);
  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  cost_graph->AddEdge(matmul2, matmul4, edge_m2_m4);
  matmul3->AddSuccEdge(edge_m3_m4);
  matmul4->AddPrevEdge(edge_m3_m4);
  cost_graph->AddEdge(matmul3, matmul4, edge_m3_m4);
}

void TestDPAlgo::ConstructStarGraph2() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m0_m2 = std::make_shared<Edge>(edge_name, matmul0, matmul2, 0, 1, false);
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m1_m3 = std::make_shared<Edge>(edge_name, matmul1, matmul3, 0, 1, false);

  matmul0->GenerateStrategies(0);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  cost_graph->AddOperator(matmul0);
  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul3);
  edge_m0_m2->InitEdgeCost();
  edge_m1_m2->InitEdgeCost();
  edge_m1_m3->InitEdgeCost();

  matmul0->AddSuccEdge(edge_m0_m2);
  matmul2->AddPrevEdge(edge_m0_m2);
  cost_graph->AddEdge(matmul0, matmul2, edge_m0_m2);
  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(matmul1, matmul2, edge_m1_m2);
  matmul1->AddSuccEdge(edge_m1_m3);
  matmul3->AddPrevEdge(edge_m1_m3);
  cost_graph->AddEdge(matmul1, matmul3, edge_m1_m3);
}

void TestDPAlgo::ConstructStarGraph3() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m2_m1 = std::make_shared<Edge>(edge_name, matmul2, matmul1, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m6 = std::make_shared<Edge>(edge_name, matmul2, matmul6, 0, 1, false);

  matmul2->GenerateStrategies(0);
  matmul1->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  matmul6->GenerateStrategies(0);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul4);
  cost_graph->AddOperator(matmul6);
  edge_m2_m1->InitEdgeCost();
  edge_m2_m4->InitEdgeCost();
  edge_m2_m6->InitEdgeCost();

  matmul2->AddSuccEdge(edge_m2_m1);
  matmul1->AddPrevEdge(edge_m2_m1);
  cost_graph->AddEdge(matmul2, matmul1, edge_m2_m1);
  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  cost_graph->AddEdge(matmul2, matmul4, edge_m2_m4);
  matmul2->AddSuccEdge(edge_m2_m6);
  matmul6->AddPrevEdge(edge_m2_m6);
  cost_graph->AddEdge(matmul2, matmul6, edge_m2_m6);
}

void TestDPAlgo::ConstructTwoSeparateGraphs2() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m0_m2 = std::make_shared<Edge>(edge_name, matmul0, matmul2, 0, 1, false);
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m1_m3 = std::make_shared<Edge>(edge_name, matmul1, matmul3, 0, 1, false);
  std::shared_ptr<Edge> edge_m4_m5 = std::make_shared<Edge>(edge_name, matmul4, matmul5, 0, 0, false);

  matmul0->GenerateStrategies(0);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  matmul5->GenerateStrategies(0);

  cost_graph->AddOperator(matmul0);
  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul3);
  cost_graph->AddOperator(matmul4);
  cost_graph->AddOperator(matmul5);

  edge_m0_m2->InitEdgeCost();
  edge_m1_m2->InitEdgeCost();
  edge_m1_m3->InitEdgeCost();
  edge_m4_m5->InitEdgeCost();

  matmul0->AddSuccEdge(edge_m0_m2);
  matmul2->AddPrevEdge(edge_m0_m2);
  cost_graph->AddEdge(matmul0, matmul2, edge_m0_m2);
  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(matmul1, matmul2, edge_m1_m2);
  matmul1->AddSuccEdge(edge_m1_m3);
  matmul3->AddPrevEdge(edge_m1_m3);
  cost_graph->AddEdge(matmul1, matmul3, edge_m1_m3);
  matmul4->AddSuccEdge(edge_m4_m5);
  matmul5->AddPrevEdge(edge_m4_m5);
  cost_graph->AddEdge(matmul4, matmul5, edge_m4_m5);
}

void TestDPAlgo::ConstructTwoSeparateSingleNodeGraph() {
  matmul0->GenerateStrategies(0);
  matmul1->GenerateStrategies(0);

  cost_graph->AddOperator(matmul0);
  cost_graph->AddOperator(matmul1);
}

void TestDPAlgo::ConstructDoubleStarGraph() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m6_m2 = std::make_shared<Edge>(edge_name, matmul6, matmul2, 0, 1, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  std::shared_ptr<Edge> edge_m3_m4 = std::make_shared<Edge>(edge_name, matmul3, matmul4, 0, 1, false);
  std::shared_ptr<Edge> edge_m4_m5 = std::make_shared<Edge>(edge_name, matmul4, matmul5, 0, 0, false);

  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  matmul5->GenerateStrategies(0);
  matmul6->GenerateStrategies(0);

  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul3);
  cost_graph->AddOperator(matmul4);
  cost_graph->AddOperator(matmul5);
  cost_graph->AddOperator(matmul6);

  edge_m1_m2->InitEdgeCost();
  edge_m6_m2->InitEdgeCost();
  edge_m2_m4->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();
  edge_m4_m5->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(matmul1, matmul2, edge_m1_m2);

  matmul6->AddSuccEdge(edge_m6_m2);
  matmul2->AddPrevEdge(edge_m6_m2);
  cost_graph->AddEdge(matmul6, matmul2, edge_m6_m2);

  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  cost_graph->AddEdge(matmul2, matmul4, edge_m2_m4);

  matmul3->AddSuccEdge(edge_m3_m4);
  matmul4->AddPrevEdge(edge_m3_m4);
  cost_graph->AddEdge(matmul3, matmul4, edge_m3_m4);

  matmul4->AddSuccEdge(edge_m4_m5);
  matmul5->AddPrevEdge(edge_m4_m5);
  cost_graph->AddEdge(matmul4, matmul5, edge_m4_m5);
}

void TestDPAlgo::ConstructTwoSeparateGraphs() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m6_m2 = std::make_shared<Edge>(edge_name, matmul6, matmul2, 0, 1, false);

  std::shared_ptr<Edge> edge_m3_m4 = std::make_shared<Edge>(edge_name, matmul3, matmul4, 0, 1, false);
  std::shared_ptr<Edge> edge_m4_m5 = std::make_shared<Edge>(edge_name, matmul4, matmul5, 0, 0, false);

  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  matmul5->GenerateStrategies(0);
  matmul6->GenerateStrategies(0);

  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul3);
  cost_graph->AddOperator(matmul4);
  cost_graph->AddOperator(matmul5);
  cost_graph->AddOperator(matmul6);

  edge_m1_m2->InitEdgeCost();
  edge_m6_m2->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();
  edge_m4_m5->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(matmul1, matmul2, edge_m1_m2);

  matmul6->AddSuccEdge(edge_m6_m2);
  matmul2->AddPrevEdge(edge_m6_m2);
  cost_graph->AddEdge(matmul6, matmul2, edge_m6_m2);

  matmul3->AddSuccEdge(edge_m3_m4);
  matmul4->AddPrevEdge(edge_m3_m4);
  cost_graph->AddEdge(matmul3, matmul4, edge_m3_m4);

  matmul4->AddSuccEdge(edge_m4_m5);
  matmul5->AddPrevEdge(edge_m4_m5);
  cost_graph->AddEdge(matmul4, matmul5, edge_m4_m5);
}

void TestDPAlgo::ConstructThreeSeparateGraphs() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m6_m2 = std::make_shared<Edge>(edge_name, matmul6, matmul2, 0, 1, false);

  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  matmul5->GenerateStrategies(0);
  matmul6->GenerateStrategies(0);

  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul3);
  cost_graph->AddOperator(matmul4);
  cost_graph->AddOperator(matmul5);
  cost_graph->AddOperator(matmul6);

  edge_m1_m2->InitEdgeCost();
  edge_m6_m2->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(matmul1, matmul2, edge_m1_m2);

  matmul6->AddSuccEdge(edge_m6_m2);
  matmul2->AddPrevEdge(edge_m6_m2);
  cost_graph->AddEdge(matmul6, matmul2, edge_m6_m2);
}

void TestDPAlgo::ConstructDiamondGraph() {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  std::shared_ptr<Edge> edge_m1_m3 = std::make_shared<Edge>(edge_name, matmul1, matmul3, 0, 1, false);
  std::shared_ptr<Edge> edge_m3_m4 = std::make_shared<Edge>(edge_name, matmul3, matmul4, 0, 1, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul3->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);
  cost_graph->AddOperator(matmul1);
  cost_graph->AddOperator(matmul2);
  cost_graph->AddOperator(matmul3);
  cost_graph->AddOperator(matmul4);
  edge_m1_m2->InitEdgeCost();
  edge_m2_m4->InitEdgeCost();
  edge_m1_m3->InitEdgeCost();
  edge_m3_m4->InitEdgeCost();

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(matmul1, matmul2, edge_m1_m2);
  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  cost_graph->AddEdge(matmul2, matmul4, edge_m2_m4);
  matmul1->AddSuccEdge(edge_m1_m3);
  matmul3->AddPrevEdge(edge_m1_m3);
  cost_graph->AddEdge(matmul1, matmul3, edge_m1_m3);
  matmul3->AddSuccEdge(edge_m3_m4);
  matmul4->AddPrevEdge(edge_m3_m4);
  cost_graph->AddEdge(matmul3, matmul4, edge_m3_m4);
}

void TestDPAlgo::ConstructMMRGraph() {
  // mm1_ptr
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};
  Shapes inputs_shape_1 = {{32, 16}, {16, 32}};
  Shapes outputs_shape_1 = {{32, 32}};
  mm1_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);
  mm1_ptr->set_outputs_type({kFloat32});

  // mm2_ptr
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpose_b_2}};
  Shapes inputs_shape_2 = {{8, 32}, {32, 32}};
  Shapes outputs_shape_2 = {{8, 32}};
  mm2_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_2, outputs_shape_2, attr_2);
  mm2_ptr->set_outputs_type({kFloat32});

  // mm3_ptr
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpose_b_3 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_3}};
  Shapes inputs_shape_3 = {{32, 32}, {32, 64}};
  Shapes outputs_shape_3 = {{32, 64}};
  mm3_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_3);
  mm3_ptr->set_outputs_type({kFloat32});

  // mm4_ptr
  ValuePtr transpose_a_4 = MakeValue(false);
  ValuePtr transpose_b_4 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_4 = {{"transpose_a", transpose_a_4}, {"transpose_b", transpose_b_4}};
  Shapes inputs_shape_4 = {{64, 32}, {32, 32}};
  Shapes outputs_shape_4 = {{64, 32}};
  mm4_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_4, outputs_shape_4, attr_4);
  mm4_ptr->set_outputs_type({kFloat32});

  // mm5_ptr
  ValuePtr transpose_a_5 = MakeValue(false);
  ValuePtr transpose_b_5 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_5 = {{"transpose_a", transpose_a_5}, {"transpose_b", transpose_b_5}};
  Shapes inputs_shape_5 = {{8, 32}, {32, 64}};
  Shapes outputs_shape_5 = {{8, 64}};
  mm5_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_5, outputs_shape_5, attr_5);
  mm5_ptr->set_outputs_type({kFloat32});

  // mm5_ptr
  ValuePtr transpose_a_6 = MakeValue(false);
  ValuePtr transpose_b_6 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_6 = {{"transpose_a", transpose_a_6}, {"transpose_b", transpose_b_6}};
  Shapes inputs_shape_6 = {{8, 64}, {64, 32}};
  Shapes outputs_shape_6 = {{8, 32}};
  mm6_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_6, outputs_shape_6, attr_6);
  mm6_ptr->set_outputs_type({kFloat32});

  ValuePtr relu = MakeValue(std::string("relu"));
  mindspore::HashMap<std::string, ValuePtr> relu_attr = {{"activation_type", relu}};

  // relu1_ptr
  Shapes relu1_inputs_shape = {{8, 32}};
  Shapes relu1_outputs_shape = {{8, 32}};
  relu1_ptr = std::make_shared<ActivationInfo>("relu_info", relu1_inputs_shape, relu1_outputs_shape, relu_attr);
  relu1_ptr->set_outputs_type({kFloat32});

  // relu2_ptr
  Shapes relu2_inputs_shape = {{32, 64}};
  Shapes relu2_outputs_shape = {{32, 64}};
  relu2_ptr = std::make_shared<ActivationInfo>("relu_info", relu2_inputs_shape, relu2_outputs_shape, relu_attr);
  relu2_ptr->set_outputs_type({kFloat32});

  // relu3_ptr
  Shapes relu3_inputs_shape = {{64, 32}};
  Shapes relu3_outputs_shape = {{64, 32}};
  relu3_ptr = std::make_shared<ActivationInfo>("relu_info", relu3_inputs_shape, relu3_outputs_shape, relu_attr);
  relu3_ptr->set_outputs_type({kFloat32});

  // relu4_ptr
  Shapes relu4_inputs_shape = {{8, 64}};
  Shapes relu4_outputs_shape = {{8, 64}};
  relu4_ptr = std::make_shared<ActivationInfo>("relu_info", relu4_inputs_shape, relu4_outputs_shape, relu_attr);
  relu4_ptr->set_outputs_type({kFloat32});

  // relu5_ptr
  Shapes relu5_inputs_shape = {{8, 32}};
  Shapes relu5_outputs_shape = {{8, 32}};
  relu5_ptr = std::make_shared<ActivationInfo>("relu_info", relu5_inputs_shape, relu5_outputs_shape, relu_attr);
  relu5_ptr->set_outputs_type({kFloat32});

  std::string edge_matmul_matmul_name = "MatMul-MatMul";
  std::string edge_matmul_relu_name = "MatMul-ReLU";
  std::string edge_relu_matmul_name = "ReLU-MatMul";

  // create edges
  edge_m1_m2 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm2_ptr, 0, 1, false);
  edge_m1_m3 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm3_ptr, 0, 0, false);
  edge_m1_m4 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm4_ptr, 0, 1, false);
  edge_m2_r1 = std::make_shared<Edge>(edge_matmul_relu_name, mm2_ptr, relu1_ptr, 0, 0, false);
  edge_m3_r2 = std::make_shared<Edge>(edge_matmul_relu_name, mm3_ptr, relu2_ptr, 0, 0, false);
  edge_m4_r3 = std::make_shared<Edge>(edge_matmul_relu_name, mm4_ptr, relu3_ptr, 0, 0, false);
  edge_r1_m5 = std::make_shared<Edge>(edge_relu_matmul_name, relu1_ptr, mm5_ptr, 0, 0, false);
  edge_r2_m5 = std::make_shared<Edge>(edge_relu_matmul_name, relu2_ptr, mm5_ptr, 0, 1, false);
  edge_m5_r4 = std::make_shared<Edge>(edge_matmul_relu_name, mm5_ptr, relu4_ptr, 0, 0, false);
  edge_r4_m6 = std::make_shared<Edge>(edge_relu_matmul_name, relu4_ptr, mm6_ptr, 0, 0, false);
  edge_r3_m6 = std::make_shared<Edge>(edge_relu_matmul_name, relu3_ptr, mm6_ptr, 0, 1, false);
  edge_m6_r5 = std::make_shared<Edge>(edge_matmul_relu_name, mm6_ptr, relu5_ptr, 0, 0, false);

  // init operators
  mm1_ptr->GenerateStrategies(0);
  mm2_ptr->GenerateStrategies(0);
  mm3_ptr->GenerateStrategies(0);
  mm4_ptr->GenerateStrategies(0);
  mm5_ptr->GenerateStrategies(0);
  mm6_ptr->GenerateStrategies(0);
  relu1_ptr->GenerateStrategies(0);
  relu2_ptr->GenerateStrategies(0);
  relu3_ptr->GenerateStrategies(0);
  relu4_ptr->GenerateStrategies(0);
  relu5_ptr->GenerateStrategies(0);

  // add operators to costgraph
  cost_graph->AddOperator(mm1_ptr);
  cost_graph->AddOperator(mm2_ptr);
  cost_graph->AddOperator(mm3_ptr);
  cost_graph->AddOperator(mm4_ptr);
  cost_graph->AddOperator(mm5_ptr);
  cost_graph->AddOperator(mm6_ptr);
  cost_graph->AddOperator(relu1_ptr);
  cost_graph->AddOperator(relu2_ptr);
  cost_graph->AddOperator(relu3_ptr);
  cost_graph->AddOperator(relu4_ptr);
  cost_graph->AddOperator(relu5_ptr);

  // init edge cost
  edge_m1_m2->InitEdgeCost();
  edge_m1_m3->InitEdgeCost();
  edge_m1_m4->InitEdgeCost();
  edge_m2_r1->InitEdgeCost();
  edge_m3_r2->InitEdgeCost();
  edge_m4_r3->InitEdgeCost();
  edge_r1_m5->InitEdgeCost();
  edge_r2_m5->InitEdgeCost();
  edge_m5_r4->InitEdgeCost();
  edge_r4_m6->InitEdgeCost();
  edge_r3_m6->InitEdgeCost();
  edge_m6_r5->InitEdgeCost();

  mm1_ptr->AddSuccEdge(edge_m1_m2);
  mm2_ptr->AddPrevEdge(edge_m1_m2);
  cost_graph->AddEdge(mm1_ptr, mm2_ptr, edge_m1_m2);

  mm1_ptr->AddSuccEdge(edge_m1_m3);
  mm3_ptr->AddPrevEdge(edge_m1_m3);
  cost_graph->AddEdge(mm1_ptr, mm3_ptr, edge_m1_m3);

  mm1_ptr->AddSuccEdge(edge_m1_m4);
  mm4_ptr->AddPrevEdge(edge_m1_m4);
  cost_graph->AddEdge(mm1_ptr, mm4_ptr, edge_m1_m4);

  mm2_ptr->AddSuccEdge(edge_m2_r1);
  relu1_ptr->AddPrevEdge(edge_m2_r1);
  cost_graph->AddEdge(mm2_ptr, relu1_ptr, edge_m2_r1);

  mm3_ptr->AddSuccEdge(edge_m3_r2);
  relu2_ptr->AddPrevEdge(edge_m3_r2);
  cost_graph->AddEdge(mm3_ptr, relu2_ptr, edge_m3_r2);

  mm4_ptr->AddSuccEdge(edge_m4_r3);
  relu3_ptr->AddPrevEdge(edge_m4_r3);
  cost_graph->AddEdge(mm4_ptr, relu3_ptr, edge_m4_r3);

  relu1_ptr->AddSuccEdge(edge_r1_m5);
  mm5_ptr->AddPrevEdge(edge_r1_m5);
  cost_graph->AddEdge(relu1_ptr, mm5_ptr, edge_r1_m5);

  relu2_ptr->AddSuccEdge(edge_r2_m5);
  mm5_ptr->AddPrevEdge(edge_r2_m5);
  cost_graph->AddEdge(relu2_ptr, mm5_ptr, edge_r2_m5);

  mm5_ptr->AddSuccEdge(edge_m5_r4);
  relu4_ptr->AddPrevEdge(edge_m5_r4);
  cost_graph->AddEdge(mm5_ptr, relu4_ptr, edge_m5_r4);

  relu4_ptr->AddSuccEdge(edge_r4_m6);
  mm6_ptr->AddPrevEdge(edge_r4_m6);
  cost_graph->AddEdge(relu4_ptr, mm6_ptr, edge_r4_m6);

  relu3_ptr->AddSuccEdge(edge_r3_m6);
  mm6_ptr->AddPrevEdge(edge_r3_m6);
  cost_graph->AddEdge(relu3_ptr, mm6_ptr, edge_r3_m6);

  mm6_ptr->AddSuccEdge(edge_m6_r5);
  relu5_ptr->AddPrevEdge(edge_m6_r5);
  cost_graph->AddEdge(mm6_ptr, relu5_ptr, edge_m6_r5);
}

void TestDPAlgo::ConstructIdentityDiamondGraph() {
  mindspore::HashMap<std::string, ValuePtr> attr = {};
  Shapes inputs_shape = {{32, 64}};
  Shapes outputs_shape = {{32, 64}};
  tmp_identity_ptr = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
  tmp_identity_ptr->set_outputs_type({kFloat32});

  // mm1_ptr
  ValuePtr transpose_a_1 = MakeValue(false);
  ValuePtr transpose_b_1 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_1 = {{"transpose_a", transpose_a_1}, {"transpose_b", transpose_b_1}};
  Shapes inputs_shape_1 = {{32, 64}, {64, 128}};
  Shapes outputs_shape_1 = {{32, 128}};
  mm1_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_1, outputs_shape_1, attr_1);
  mm1_ptr->set_outputs_type({kFloat32});

  // mm2_ptr
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_2 = {{"transpose_a", transpose_a_2}, {"transpose_b", transpose_b_2}};
  Shapes inputs_shape_2 = {{128, 32}, {32, 64}};
  Shapes outputs_shape_2 = {{128, 64}};
  mm2_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_2, outputs_shape_2, attr_2);
  mm2_ptr->set_outputs_type({kFloat32});

  // mm3_ptr
  ValuePtr transpose_a_3 = MakeValue(false);
  ValuePtr transpose_b_3 = MakeValue(false);
  mindspore::HashMap<std::string, ValuePtr> attr_3 = {{"transpose_a", transpose_a_3}, {"transpose_b", transpose_b_3}};
  Shapes inputs_shape_3 = {{32, 128}, {128, 64}};
  Shapes outputs_shape_3 = {{32, 64}};
  mm3_ptr = std::make_shared<MatMulInfo>("matmul_info", inputs_shape_3, outputs_shape_3, attr_3);
  mm3_ptr->set_outputs_type({kFloat32});

  // create edges
  std::string edge_matmul_matmul_name = "MatMul-MatMul";
  std::string edge_iden_matmul_name = "TmpIdentity-MatMul";
  edge_i1_m1 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr, mm1_ptr, 0, 0, false, true);
  edge_i1_m2 = std::make_shared<Edge>(edge_iden_matmul_name, tmp_identity_ptr, mm2_ptr, 0, 1, false, true);
  edge_m1_m3 = std::make_shared<Edge>(edge_matmul_matmul_name, mm1_ptr, mm3_ptr, 0, 0, false);
  edge_m2_m3 = std::make_shared<Edge>(edge_matmul_matmul_name, mm2_ptr, mm3_ptr, 0, 1, false);

  // init operators
  tmp_identity_ptr->GenerateStrategies(0);
  mm1_ptr->GenerateStrategies(0);
  mm2_ptr->GenerateStrategies(0);
  mm3_ptr->GenerateStrategies(0);

  // add operators to costgraph
  cost_graph->AddOperator(tmp_identity_ptr);
  cost_graph->AddOperator(mm1_ptr);
  cost_graph->AddOperator(mm2_ptr);
  cost_graph->AddOperator(mm3_ptr);

  // init edge cost
  edge_i1_m1->InitEdgeCost();
  edge_i1_m2->InitEdgeCost();
  edge_m1_m3->InitEdgeCost();
  edge_m2_m3->InitEdgeCost();

  // add edges to costgraph
  tmp_identity_ptr->AddSuccEdge(edge_i1_m1);
  mm1_ptr->AddPrevEdge(edge_i1_m1);
  cost_graph->AddEdge(tmp_identity_ptr, mm1_ptr, edge_i1_m1);

  tmp_identity_ptr->AddSuccEdge(edge_i1_m2);
  mm2_ptr->AddPrevEdge(edge_i1_m2);
  cost_graph->AddEdge(tmp_identity_ptr, mm2_ptr, edge_i1_m2);

  mm1_ptr->AddSuccEdge(edge_m1_m3);
  mm3_ptr->AddPrevEdge(edge_m1_m3);
  cost_graph->AddEdge(mm1_ptr, mm3_ptr, edge_m1_m3);

  mm2_ptr->AddSuccEdge(edge_m2_m3);
  mm3_ptr->AddPrevEdge(edge_m2_m3);
  cost_graph->AddEdge(mm2_ptr, mm3_ptr, edge_m2_m3);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructTwoLargeMatMul) {
  ConstructTwoLargeMatMul();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
  ASSERT_EQ(cost_graph->InitSelectedStrategy(), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructBatmanGraph) {
  ConstructBatmanGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
  ASSERT_EQ(cost_graph->InitSelectedStrategy(), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructTriangleGraph) {
  ConstructTriangleGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructTriangleGraph2) {
  ConstructTriangleGraph2();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructStarGraph2) {
  ConstructStarGraph2();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructStarGraph3) {
  ConstructStarGraph3();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructTwoSeparateGraphs2) {
  ConstructTwoSeparateGraphs2();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructTwoSeparateSingleNodeGraph) {
  ConstructTwoSeparateSingleNodeGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructThreeSeparateGraphs) {
  ConstructThreeSeparateGraphs();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_ConstructTwoSeparateGraphs) {
  ConstructTwoSeparateGraphs();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_GetStrategy) {
  ConstructDiamondGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_GetStrategy_for_MMR_graph) {
  ConstructMMRGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_GetStrategy_for_IdentityDiamondGraph) {
  ConstructIdentityDiamondGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_GetStrategy_for_StarGraph) {
  ConstructStarGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);
}

TEST_F(TestDPAlgo, DISABLED_test_GetStrategy_for_DoubleStarGraph) {
  ConstructDoubleStarGraph();
  ASSERT_EQ(GetStrategy(cost_graph), SUCCESS);

  for (auto &op : cost_graph->GetOperators()) {
    StrategyPtr s_strategy = op->selected_strategy();
    Dimensions strategy_0 = s_strategy->GetInputDim()[0];
    Dimensions strategy_1 = s_strategy->GetInputDim()[1];

    std::string string_strategy_0 = "[";
    for (size_t i = 0; i < strategy_0.size(); ++i) {
      string_strategy_0 += std::to_string(strategy_0[i]) + ", ";
    }
    string_strategy_0 += "]";

    std::string string_strategy_1 = "[";
    for (size_t i = 0; i < strategy_1.size(); ++i) {
      string_strategy_1 += std::to_string(strategy_1[i]) + ", ";
    }
    string_strategy_1 += "]";

    MS_LOG(INFO) << "" << op->name() << " selected strategy: " << string_strategy_0 << ", " << string_strategy_1;
  }
}
}  // namespace parallel
}  // namespace mindspore
