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

#include "graph_kernel/kernel_packet/kernel_packet_common_test_suite.h"

namespace mindspore::graphkernel::test {
/// Feature: KernelPacket
/// Description: the "Reshape"'s value-depend nodes have a "Depend"
/// Expectation: the virtual node should not be fused.
TEST_F(TestKernelPacket, no_virtual_node) {
  ConstructGraph gb;
  auto p1 = gb.NewTensorInput("p1", kFloat32, {-1, -1, -1, 1152});
  auto p2 = gb.NewCNodeWithBuildInfo("TensorToScalar", {gb.NewTensorInput("p2", kInt64, {})});
  auto p3_1 = gb.NewTensorInput("p3_1", kFloat32, {16, 16});
  auto p3_2 = gb.NewTensorInput("p3_2", kFloat32, {16, 32});
  auto p3 = gb.NewCNode("MakeTuple", {p3_1, p3_2});
  auto p4 = gb.NewCNodeWithBuildInfo("TensorToScalar", {gb.NewTensorInput("p4", kInt64, {})});
  auto n0 = gb.NewCNodeWithBuildInfo("Shape", {p1});
  auto n1 = gb.NewCNodeWithBuildInfo("RealTupleGetItem", {n0, gb.NewValueNode(MakeValue<int64_t>(0))});
  auto n2 = gb.NewCNode("Depend", {p2, p3});
  p4 = gb.NewCNode("Depend", {p4, p3});
  auto n3 = gb.NewCNodeWithBuildInfo("ScalarMul", {n2, p4});
  auto n4 = gb.NewCNodeWithBuildInfo("RealMakeTuple", {n1, n3, gb.NewValueNode(MakeValue<int64_t>(1152))});
  auto n5 = gb.NewCNodeWithBuildInfo("Reshape", {p1, n4});
  gb.SetOutput(n5);
  RunPass(gb.GetGraph(), {std::make_shared<packet::SymbolEngineExtender>()});
  auto nodes = TopoSort(gb.GetGraph()->output());
  auto depend_node_count = std::count_if(
    nodes.begin(), nodes.end(), [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimDepend); });
  EXPECT_EQ(depend_node_count, 2);
  auto packet_node = std::count_if(nodes.begin(), nodes.end(),
                                   [](const AnfNodePtr &node) { return GetCNodeFuncGraph(node) != nullptr; });
  EXPECT_EQ(packet_node, 1);
}
}  // namespace mindspore::graphkernel::test
