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
/// Description: fuse the nop_op "Reshape"
/// Expectation: the "Reshape" is fused and set to nop op.
TEST_F(TestKernelPacket, nop_node_1) {
  ConstructGraph gb;
  auto p1 = gb.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = gb.NewTensorInput("p2", kFloat32, {-1, -1});
  auto shape = gb.NewCNodeWithBuildInfo("Shape", {p2});
  auto out = gb.NewCNodeWithBuildInfo("Reshape", {p1, shape});
  gb.SetOutput(out);
  RunPass(gb.GetGraph(), {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  auto nodes = TopoSort(gb.GetGraph()->output());
  auto packet_node_iter = std::find_if(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    return node->isa<CNode>() && common::AnfAlgo::HasNodeAttr(kAttrKernelPacketNode, node->cast<CNodePtr>());
  });
  ASSERT_NE(packet_node_iter, nodes.end());
  auto packet_node = (*packet_node_iter)->cast<CNodePtr>();
  ASSERT_NE(packet_node, nullptr);
  EXPECT_TRUE(common::AnfAlgo::HasNodeAttr(kAttrNopOp, packet_node));

  auto sub_fg = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(packet_node, kAttrFuncGraph);
  ASSERT_NE(sub_fg, nullptr);
  // check the reshape's first input is the first input of packet node.
  auto reshape = sub_fg->output()->cast<CNodePtr>();
  ASSERT_NE(reshape, nullptr);
  ASSERT_EQ(GetCNodePrimitive(reshape)->name(), "Reshape");
  auto param = reshape->input(1);
  ASSERT_TRUE(param->isa<Parameter>());
  EXPECT_EQ(param, sub_fg->parameters()[0]);
  EXPECT_EQ(packet_node->input(1), p1);
}
}  // namespace mindspore::graphkernel::test
