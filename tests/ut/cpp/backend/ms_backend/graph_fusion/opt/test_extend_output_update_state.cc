/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <map>
#include <string>
#include "ir/tensor_new.h"
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/ms_backend/graph_fusion/core/update_state_formatter.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::graphkernel::test {
FuncGraphPtr ConstructSubgraph() {
  test::ConstructGraph c;
  auto p1 = c.NewTensorInput("p1", kFloat32, {1, 8192, 4096});
  auto p2 = c.NewTensorInput("p2", kBFloat16, {1, 8192, 4096});
  auto op0 = c.NewCNodeWithBuildInfo("Cast", {p2, c.NewValueNode<int64_t>(43)});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {op0, p1});
  auto op2 = c.NewCNodeWithBuildInfo("Cast", {op1, c.NewValueNode<int64_t>(45)});
  auto op3 = c.NewCNodeWithBuildInfo("Cast", {op1, c.NewValueNode<int64_t>(45)});
  auto op4 = c.NewCNodeWithBuildInfo("MakeTuple", {op2, op3});
  c.SetOutput(op4);
  return c.GetGraph();
}

FuncGraphPtr ConstructMainGraph(bool is_depend) {
  test::ConstructGraph c;
  auto p1 = c.NewTensorInput("p1", kFloat32, {1, 8192, 4096});
  auto p2 = c.NewTensorInput("p2", kBFloat16, {1, 8192, 4096});
  auto p3 = c.NewTensorInput("p3", kFloat32, {1, 8192, 4096});
  auto sub_fg = ConstructSubgraph();
  sub_fg->set_attr("graph_kernel", MakeValue<std::string>("GraphKernel_Cast_Add_Cast_Cast_fusion"));
  std::vector<AnfNodePtr> new_inputs = {NewValueNode(sub_fg), p1, p2};
  auto call = c.GetGraph()->NewCNode(new_inputs);
  call->set_abstract(sub_fg->output()->abstract());
  auto item0 = c.NewCNode("TupleGetItem", {call, c.NewValueNode<int64_t>(0)});
  auto out0 = c.NewCNodeWithBuildInfo("Reshape", {item0, c.NewValueNode<std::vector<int64_t>>({1, 8192, 1, 4096})});
  auto item1 = c.NewCNode("TupleGetItem", {call, c.NewValueNode<int64_t>(1)});
  CNodePtr depend;
  if (is_depend) {
    depend = c.NewCNode("Depend", {p3, item1});
  } else {
    depend = c.NewCNode("UpdateState", {p3, item1});
  }
  auto mt = c.NewCNode("MakeTuple", {out0, depend});
  c.SetOutput(mt);
  return c.GetGraph();
}

/// Feature: Test optimize assign pass
/// Description: ExtendOutputForUpdateState, the next node of subgraph is Depend
/// Expectation: Do not optimize the depend
TEST_F(GraphKernelCommonTestSuite, opt_depend) {
  auto fg = ConstructMainGraph(true);
  RunPass(fg, {std::make_shared<graphkernel::ExtendOutputForUpdateState>()});
  auto gk_nodes = GetAllGKNodes(fg);
  ASSERT_EQ(gk_nodes.size(), 1);
  ASSERT_TRUE(IsPrimitiveCNode(GetCNodeFuncGraph(gk_nodes[0])->output(), prim::kPrimMakeTuple));
}

/// Feature: Test optimize assign pass
/// Description: ExtendOutputForUpdateState, the next node of subgraph is UpdateState
/// Expectation: Optimize the update_state
TEST_F(GraphKernelCommonTestSuite, opt_update_state) {
  auto fg = ConstructMainGraph(false);
  RunPass(fg, {std::make_shared<graphkernel::ExtendOutputForUpdateState>()});
  auto gk_nodes = GetAllGKNodes(fg);
  ASSERT_EQ(gk_nodes.size(), 1);
  ASSERT_FALSE(IsPrimitiveCNode(GetCNodeFuncGraph(gk_nodes[0])->output(), prim::kPrimMakeTuple));
}

FuncGraphPtr ConstructSubGraph_two_same_getitem_idx() {
  test::ConstructGraph c;
  auto p1 = c.NewTensorInput("p1", kFloat32, {64128, 4096});
  auto p2 = c.NewTensorInput("p2", kFloat32, {64128, 4096});
  auto p3 = c.NewTensorInput("p3", kFloat32, {64128, 4096});
  auto const1 = c.NewValueNode(tensor::from_scalar((float)(0.00099987), kFloat32));
  auto node0 = c.NewCNodeWithBuildInfo("Mul", {p2, p2});
  auto node1 = c.NewCNodeWithBuildInfo("Mul", {const1, node0});
  auto node2 = c.NewCNodeWithBuildInfo("Add", {p1, node1});
  auto node3 = c.NewCNodeWithBuildInfo("Assign", {p3, node2});
  auto node4 = c.NewCNodeWithBuildInfo("MakeTuple", {node2, node3});
  c.SetOutput(node4);
  return c.GetGraph();
}

FuncGraphPtr ConstructGraph_two_same_getitem_idx() {
  test::ConstructGraph c;
  auto p1 = c.NewTensorInput("p1", kFloat32, {64128, 4096});
  auto p2 = c.NewTensorInput("p2", kFloat32, {64128, 4096});
  auto p3 = c.NewTensorInput("p3", kFloat32, {64128, 4096});
  auto sub_fg = ConstructSubGraph_two_same_getitem_idx();
  sub_fg->set_attr("graph_kernel", MakeValue<std::string>("GraphKernel_Addcmul_Assign_fusion"));
  auto call = c.GetGraph()->NewCNode({NewValueNode(sub_fg), p1, p2, p3});
  call->set_abstract(sub_fg->output()->abstract());
  auto item_a = c.NewCNode("TupleGetItem", {call, c.NewValueNode<int64_t>(1)});
  auto out0 = c.NewCNode("UpdateState", {c.NewValueNode(kUMonad), item_a});
  auto item_b = c.NewCNode("TupleGetItem", {call, c.NewValueNode<int64_t>(1)});
  auto out1 = c.NewCNode("UpdateState", {c.NewValueNode(kUMonad), item_b});
  auto mt = c.NewCNode("MakeTuple", {out0, out1});
  c.SetOutput(mt);
  return c.GetGraph();
}

/// Feature: ExtendOutputForUpdateState
/// Description: Issue #IBUN8R. There are two getitem of same index for a subgraph.
/// Expectation: After this pass, the output_0 of subgraph and getitem nodes are eliminated.
TEST_F(GraphKernelCommonTestSuite, two_same_getitem_idx) {
  SetGraphKernelFlags("--enable_cluster_ops=Assign");
  auto fg = ConstructGraph_two_same_getitem_idx();
  RunPass(fg, {std::make_shared<graphkernel::ExtendOutputForUpdateState>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 1);
  auto nodes = GetAllNodes(fg);
  auto getitem_count = std::count_if(
    nodes.begin(), nodes.end(), [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimTupleGetItem); });
  ASSERT_EQ(getitem_count, 0);
}
}  // namespace mindspore::graphkernel::test
