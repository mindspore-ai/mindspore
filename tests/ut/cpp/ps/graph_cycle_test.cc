/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "ir/anf.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "primitive/math_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/array_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ps {
class TestGraphCycle : public UT::Common {
 public:
  TestGraphCycle() = default;
  virtual ~TestGraphCycle() = default;

  void SetUp() override {}
  void TearDown() override {}
};

// Feature: Graph cycle detection and recovery
// Description: Test function FindGraphCircle.
// Expectation: Found graph circle as expected.
TEST_F(TestGraphCycle, TestSingleGraphCycleDetect) {
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  auto input1 = graph->add_parameter();
  auto input2 = graph->add_parameter();
  auto node1 = graph->NewCNode({NewValueNode(prim::kPrimAdd), input1, input2});
  auto node2 = graph->NewCNode({NewValueNode(prim::kPrimSub), node1, input2});
  graph->set_output(node2);

  // After convert, the graph is:
  //   %1 = input1 + %2
  //   %2 = %1 - input2
  auto mng = Manage(graph, false);
  mng->SetEdge(node1, 2, node2);

  const auto &circle_nodes = circle_handler::FindGraphCircle(graph);
  ASSERT_TRUE(circle_nodes.size() == 2);
}

// Feature: Graph cycle detection and recovery
// Description: Test function FindGraphCircle.
// Expectation: Found graph circle as expected.
TEST_F(TestGraphCycle, TestSingleGraphCycleWithDependDetect) {
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  auto input1 = graph->add_parameter();
  auto input2 = graph->add_parameter();
  auto node1 = graph->NewCNode({NewValueNode(prim::kPrimAdd), input1, NewValueNode(1)});
  auto node2 = graph->NewCNode({NewValueNode(prim::kPrimDepend), input2, node1});
  auto node3 = graph->NewCNode({NewValueNode(prim::kPrimSub), node2, NewValueNode(1)});
  graph->set_output(node3);

  // After convert, the graph is:
  //   %1 = Depend(a, %4)
  //   %2 = %1 + 1
  //   %3 = Depend(b, %2)
  //   %4 = %3 - 1
  auto mng = Manage(graph, false);
  auto new_depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), input1, node3});
  mng->SetEdge(node1, 1, new_depend);

  const auto &circle_nodes = circle_handler::FindGraphCircle(graph);
  ASSERT_TRUE(circle_nodes.size() == 4);
}

// Feature: Graph cycle detection and recovery
// Description: Test function RevertDependNode.
// Expectation: Found graph circle as expected.
TEST_F(TestGraphCycle, TestSingleGraphCycleWithDependRecovery) {
  common::SetEnv("MS_DEV_ENABLE_PASS_CIRCLE_RECOVERY", "1");
  FuncGraphPtr graph = std::make_shared<FuncGraph>();
  auto input1 = graph->add_parameter();
  auto input2 = graph->add_parameter();
  auto node1 = graph->NewCNode({NewValueNode(prim::kPrimAdd), input1, NewValueNode(1)});
  auto node2 = graph->NewCNode({NewValueNode(prim::kPrimDepend), input2, node1});
  auto node3 = graph->NewCNode({NewValueNode(prim::kPrimSub), node2, NewValueNode(1)});
  graph->set_output(node3);

  circle_handler::SetAttrToDepend(graph);

  // After convert, the graph is:
  //   %1 = Depend(a, %4)
  //   %2 = %1 + 1
  //   %3 = Depend(b, %2)
  //   %4 = %3 - 1
  auto mng = Manage(graph, false);
  auto new_depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), input1, node3});
  mng->SetEdge(node1, 1, new_depend);

  const auto &circle_nodes = circle_handler::FindGraphCircle(graph);
  ASSERT_TRUE(circle_nodes.size() == 4);
  bool success = circle_handler::RevertDependNode(graph, mng);
  ASSERT_TRUE(success);
}
}  // namespace ps
}  // namespace mindspore