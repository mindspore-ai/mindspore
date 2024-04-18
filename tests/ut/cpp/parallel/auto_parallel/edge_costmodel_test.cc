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
#include "ir/dtype/number.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "frontend/parallel/ops_info/matmul_info.h"

namespace mindspore {
namespace parallel {

using MatMulInfoPtr = std::shared_ptr<MatMulInfo>;
class TestEdgeCostModel : public UT::Common {
 public:
  TestEdgeCostModel() {
    matmul1 = nullptr;
    matmul2 = nullptr;
    matmul3 = nullptr;
    matmul4 = nullptr;
    matmul5 = nullptr;
  }
  void SetUp();
  void TearDown() {}
  MatMulInfoPtr matmul1;
  MatMulInfoPtr matmul2;
  MatMulInfoPtr matmul3;
  MatMulInfoPtr matmul4;
  MatMulInfoPtr matmul5;
};

void TestEdgeCostModel::SetUp() {
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

TEST_F(TestEdgeCostModel, DISABLED_test_InitEdgeCost) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  ASSERT_EQ(edge_m1_m2->InitEdgeCost(), SUCCESS);
}

TEST_F(TestEdgeCostModel, DISABLED_test_OpEliminationSetNewCost) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m2 = std::make_shared<Edge>(edge_name, matmul1, matmul2, 0, 0, false);
  std::shared_ptr<Edge> edge_m2_m4 = std::make_shared<Edge>(edge_name, matmul2, matmul4, 0, 0, false);
  matmul1->GenerateStrategies(0);
  matmul2->GenerateStrategies(0);
  matmul4->GenerateStrategies(0);

  matmul1->AddSuccEdge(edge_m1_m2);
  matmul2->AddPrevEdge(edge_m1_m2);
  matmul2->AddSuccEdge(edge_m2_m4);
  matmul4->AddPrevEdge(edge_m2_m4);
  ASSERT_EQ(edge_m1_m2->InitEdgeCost(), SUCCESS);
  ASSERT_EQ(edge_m2_m4->InitEdgeCost(), SUCCESS);
  std::shared_ptr<Edge> new_edge = std::make_shared<Edge>(edge_name, matmul1, matmul4, 0, 0, false);
  new_edge->set_pre_op_output(edge_m1_m2->prev_op_output());
  new_edge->set_next_op_input(edge_m2_m4->next_op_input());
  new_edge->OpEliminationSetNewCost(edge_m1_m2, matmul2, edge_m2_m4);
}

TEST_F(TestEdgeCostModel, DISABLED_test_EdgeEliminationSetNewCost) {
  std::string edge_name = "MatMul-MatMul";
  std::shared_ptr<Edge> edge_m1_m5 = std::make_shared<Edge>(edge_name, matmul1, matmul5, 0, 0, false);
  std::shared_ptr<Edge> edge_m1_m5_2 = std::make_shared<Edge>(edge_name, matmul1, matmul5, 0, 1, false);

  matmul1->GenerateStrategies(0);
  matmul5->GenerateStrategies(0);
  matmul1->AddSuccEdge(edge_m1_m5);
  matmul1->AddSuccEdge(edge_m1_m5_2);
  matmul5->AddPrevEdge(edge_m1_m5);
  matmul5->AddPrevEdge(edge_m1_m5_2);
  ASSERT_EQ(edge_m1_m5->InitEdgeCost(), SUCCESS);
  ASSERT_EQ(edge_m1_m5_2->InitEdgeCost(), SUCCESS);

  std::vector<std::shared_ptr<Edge>> edges;
  edges.push_back(edge_m1_m5);
  edges.push_back(edge_m1_m5_2);
  std::vector<size_t> output_indexs, input_indexs;
  output_indexs.push_back(0);
  input_indexs.push_back(0);
  input_indexs.push_back(1);
  std::shared_ptr<Edge> new_edge =
    std::make_shared<Edge>(edge_name, matmul1, matmul5, output_indexs, input_indexs, true);
  new_edge->set_pre_op_output(edges[0]->prev_op_output());
  new_edge->set_next_op_input(edges[0]->next_op_input());
  new_edge->EdgeEliminationSetNewCost(matmul1, edges, matmul5);
}

}  // namespace parallel
}  // namespace mindspore
