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
#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"
#include <memory>
#include "ir/value.h"

namespace mindspore {
namespace parallel {
#define ARRAY_A 3000  // also 'I' :height of the first input tensor
#define ARRAY_B 1000  // also 'K' :used by both input tensor
#define ARRAY_C 4000  // also 'J' :width of the first input tensor

class TestPartition : public UT::Common {
 public:
  void Create(std::shared_ptr<Graph> graph, int node_num, std::vector<int64_t> edge_head,
              std::vector<int64_t> edge_tail);
  void InitEdge(std::shared_ptr<Graph> graph, int vHead, int vTail);
  void InitNode(std::shared_ptr<Graph> graph, int num_node);
  TensorParam *MakeTensor(int n, int c, int h, int w);
  std::shared_ptr<Graph> MakeMatMulData(int numNode);
};

// Local function to create test input graph with nodes
void TestPartition::Create(std::shared_ptr<Graph> graph, int node_num, std::vector<int64_t> edge_head,
                           std::vector<int64_t> edge_tail) {
  TestPartition::InitNode(graph, node_num);
  unsigned int edge_num = edge_head.size();
  if (edge_num != edge_tail.size()) {
    exit(1);
  };

  for (unsigned int i = 0; i < edge_num; i++) {
    TestPartition::InitEdge(graph, edge_head[i], edge_tail[i]);
  };
}

// Local function for Create() to crate Node
void TestPartition::InitNode(std::shared_ptr<Graph> graph, int num_node) {
  Graph::NodeType NewNode;
  for (int i = 0; i < num_node; i++) {
    graph->nodes.push_back(NewNode);
    std::stringstream ss;
    ss << 'N' << i;
    graph->nodes[i].name = ss.str();
    graph->nodes[i].info = kConstant;
  };
}

// Local function for Create() to crate Edge
void TestPartition::InitEdge(std::shared_ptr<Graph> graph, int vHead, int vTail) {
  graph->nodes[vHead].node_out.push_back(vTail);
  graph->nodes[vTail].node_in.push_back(vHead);
}

// Local function for Create() to crate Tensor
TensorParam *TestPartition::MakeTensor(int n, int c, int h, int w) {
  TensorParam *p_tensor = new TensorParam;
  p_tensor->tensor_type = kFloat32;
  p_tensor->tensor_shape.shape_n = n;
  p_tensor->tensor_shape.shape_c = c;
  p_tensor->tensor_shape.shape_h = h;
  p_tensor->tensor_shape.shape_w = w;

  return p_tensor;
};

// Local function for Create() to create MatMul Operator
// @numNode include Tensor and Operator, for example 4(1 Input Tensor, 1 Input Tensor, 1 Operator, 1 Output Tensor)
std::shared_ptr<Graph> TestPartition::MakeMatMulData(int numNode) {
  // Build Edges
  int edgeNum = 0;
  constexpr int INTERVAL = 2;
  if (numNode % INTERVAL == 0 && numNode != 0) {
    edgeNum = numNode - INTERVAL;
  } else if (numNode % INTERVAL == 1) {
    edgeNum = numNode - 1;
  } else {
    edgeNum = 0;
  };

  std::vector<int64_t> edgeHead(edgeNum);  // int edgeHead[8] = {0,2,4,6,1,3,5,7};
  std::vector<int64_t> edgeTail(edgeNum);  // int edgeTail[8] = {2,4,6,8,2,4,6,8};

  for (int i = 0; i < edgeNum; i++) {
    edgeHead[i] = i;
    if (i % INTERVAL == 0) {
      edgeTail[i] = i + INTERVAL;
    } else {
      edgeTail[i] = i + 1;
    };
  };

  // Create graph
  std::shared_ptr<Graph> graph(new Graph);
  TestPartition::Create(graph, numNode, edgeHead, edgeTail);

  // Add Node information.
  for (int i = 0; i < numNode; i++) {
    if (0 == i) {
      graph->nodes[i].info = InfoType::kConstant;
      TensorParam *p_tensor_out = new TensorParam;
      p_tensor_out->tensor_type = kFloat32;
      p_tensor_out->tensor_shape.shape_w = ARRAY_B;
      p_tensor_out->tensor_shape.shape_h = ARRAY_A;

      graph->nodes[i].tensor_parm = *p_tensor_out;

    } else if (0 == i % 4) {
      graph->nodes[i].info = InfoType::kApplication;
      graph->nodes[i].apply.op_type = OperatorType::kRecMatMul;

      TensorParam *p_tensor0 = new TensorParam;
      p_tensor0->tensor_type = kFloat32;
      p_tensor0->tensor_shape.shape_w = ARRAY_C;
      p_tensor0->tensor_shape.shape_h = ARRAY_A;

      TensorParam *p_tensor1 = new TensorParam;
      p_tensor1->tensor_type = kFloat32;
      p_tensor1->tensor_shape.shape_w = ARRAY_B;
      p_tensor1->tensor_shape.shape_h = ARRAY_C;

      TensorParam *p_tensor_out = new TensorParam;
      p_tensor_out->tensor_type = kFloat32;
      p_tensor_out->tensor_shape.shape_w = ARRAY_B;
      p_tensor_out->tensor_shape.shape_h = ARRAY_A;

      graph->nodes[i].apply.arguments[0] = *p_tensor0;
      graph->nodes[i].apply.arguments[1] = *p_tensor1;
      graph->nodes[i].tensor_parm = *p_tensor_out;

    } else if (1 == i % 4) {
      graph->nodes[i].info = InfoType::kConstant;

      TensorParam *p_tensor_out = new TensorParam;
      p_tensor_out->tensor_type = kFloat32;
      p_tensor_out->tensor_shape.shape_w = ARRAY_C;
      p_tensor_out->tensor_shape.shape_h = ARRAY_B;

      graph->nodes[i].tensor_parm = *p_tensor_out;

    } else if (2 == i % 4) {
      graph->nodes[i].info = InfoType::kApplication;
      graph->nodes[i].apply.op_type = OperatorType::kRecMatMul;

      TensorParam *p_tensor0 = new TensorParam;
      p_tensor0->tensor_type = kFloat32;
      p_tensor0->tensor_shape.shape_w = ARRAY_B;
      p_tensor0->tensor_shape.shape_h = ARRAY_A;

      TensorParam *p_tensor1 = new TensorParam;
      p_tensor1->tensor_type = kFloat32;
      p_tensor1->tensor_shape.shape_w = ARRAY_C;
      p_tensor1->tensor_shape.shape_h = ARRAY_B;

      TensorParam *p_tensor_out = new TensorParam;
      p_tensor_out->tensor_type = kFloat32;
      p_tensor_out->tensor_shape.shape_w = ARRAY_C;
      p_tensor_out->tensor_shape.shape_h = ARRAY_A;

      graph->nodes[i].apply.arguments[0] = *p_tensor0;
      graph->nodes[i].apply.arguments[1] = *p_tensor1;
      graph->nodes[i].tensor_parm = *p_tensor_out;

    } else if (3 == i % 4) {
      graph->nodes[i].info = InfoType::kConstant;

      TensorParam *p_tensor_out = new TensorParam;
      p_tensor_out->tensor_type = kFloat32;
      p_tensor_out->tensor_shape.shape_w = ARRAY_B;
      p_tensor_out->tensor_shape.shape_h = ARRAY_C;

      graph->nodes[i].tensor_parm = *p_tensor_out;
    };
  };
  return graph;
};

TEST_F(TestPartition, test_GetWeights) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  double wop1 = GetWeights(graph->nodes[2]);
  double wop2 = GetWeights(graph->nodes[4]);
  double wop3 = GetWeights(graph->nodes[6]);
  double wop4 = GetWeights(graph->nodes[8]);
  ASSERT_GE(wop1, wop2);
  ASSERT_GE(wop2, wop3);
  ASSERT_GE(wop3, wop4);
}

TEST_F(TestPartition, test_SortByWeight) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  std::vector<size_t> result = SortByWeight(graph);
  ASSERT_GE(result.at(0), result.at(1));
  ASSERT_GE(result.at(1), result.at(2));
  ASSERT_GE(result.at(2), result.at(3));
}

TEST_F(TestPartition, test_SortByWeight2) {
  std::shared_ptr<Graph> graph = MakeMatMulData(5);
  std::vector<size_t> result = SortByWeight(graph);
  ASSERT_GE(result.at(0), result.at(1));
}

TEST_F(TestPartition, test_PartitionNode) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  // node 2 is the first kRecMatMul Operator
  Graph::NodeType node2 = graph->nodes[2];
  std::vector<std::pair<std::string, StrategyRec>> nameToStrategy;
  bool isTraining = true;
  StrategyRec str = PartitionNode(node2, nameToStrategy, graph, isTraining);
  ASSERT_EQ(str.outputTensor.str_h, 0.5);
  ASSERT_EQ(str.outputTensor.str_w, 1);
}

TEST_F(TestPartition, test_PartitionForAllDevices) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  double device_memory = 1024.0 * 1024.0 * 1024.0 * 16.0;
  bool isTraining = true;
  ASSERT_EQ(PartitionForAllDevices(1024, device_memory, graph, isTraining), SUCCESS);
}

TEST_F(TestPartition, test_PartitionForAllDevices2) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  double device_memory = 1024.0 * 1024.0 * 1024.0 * 16.0;
  bool isTraining = true;
  ASSERT_EQ(PartitionForAllDevices(2, device_memory, graph, isTraining), SUCCESS);
}

// Negative case: partition on 0 device
TEST_F(TestPartition, test_PartitionForAllDevices0) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  double device_memory = 1024.0 * 1024.0 * 1024.0 * 16.0;
  bool isTraining = true;
  // Throw Exception "Number of devices can't be 0"
  EXPECT_ANY_THROW(PartitionForAllDevices(0, device_memory, graph, isTraining));
}

TEST_F(TestPartition, test_ApplyStrToTensor) {
  std::shared_ptr<Graph> graph = MakeMatMulData(9);
  std::vector<std::pair<std::string, StrategyRec>> nameToStrategy;
  bool isTraining = true;
  graph->nodes[4].apply.str = PartitionNode(graph->nodes[4], nameToStrategy, graph, isTraining);
  auto h_str = graph->nodes[4].apply.str.outputTensor.str_h;
  auto w_str = graph->nodes[4].apply.str.outputTensor.str_w;

  Graph::NodeType n_node = ApplyStrToTensor(graph->nodes[4]);
  auto h_node = n_node.tensor_parm.tensor_str.str_h;
  auto w_node = n_node.tensor_parm.tensor_str.str_w;
  ASSERT_EQ(h_str, h_node);
  ASSERT_EQ(w_str, w_node);
}
}  // namespace parallel
}  // namespace mindspore
