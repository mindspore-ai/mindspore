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

#include "backend/ms_backend/graph_fusion/kernel_packet/kernel_packet_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "common/mockcpp.h"

namespace mindspore::graphkernel::packet {
bool IsOnlyOneUser(const AnfNodePtr &node);
}

namespace mindspore::graphkernel::test {
FuncGraphPtr ConstructGraph_basic() {
  ConstructGraph gb;
  auto p1 = gb.NewTensorInput("p1", kFloat32, {-1, -1});
  auto w = gb.NewTensorInput("w", kFloat32, {-1, -1});
  auto p2 = gb.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = gb.NewTensorInput("p3", kFloat32, {-1, -1});
  auto trans = gb.NewValueNode(MakeValue<bool>(false));
  auto matmul = gb.NewCNodeWithBuildInfo("MatMul", {p1, w, trans, trans});
  auto add = gb.NewCNodeWithBuildInfo("Add", {matmul, p2});
  auto shape = gb.NewCNodeWithBuildInfo("Shape", {add});
  auto reshape = gb.NewCNodeWithBuildInfo("Reshape", {p3, shape});
  gb.SetOutput(reshape);
  return gb.GetGraph();
}

/// Feature: KernelPacket
/// Description: all device nodes(without dvm op) have only one user, and they are only shape depended.
/// Expectation: fuse all nodes in kernel packet
TEST_F(TestKernelPacket, only_depend_shape_1_basic) {
  auto fg = ConstructGraph_basic();
  RunPass(fg, {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  auto nodes = GetAllCNodes(fg);
  ASSERT_EQ(nodes.size(), 1);
  NodeShapeVector realshape = {
    {fg->parameters()[0], {16, 8}},   // p1
    {fg->parameters()[1], {8, 1}},    // w
    {fg->parameters()[2], {1, 32}},   // p2
    {fg->parameters()[3], {128, 4}},  // p3
  };
  ASSERT_TRUE(InferPacketNode(nodes[0], realshape));
  auto ret_shape = mindspore::symshape::QueryShape(nodes[0]->abstract());
  UT_CHECK_NULL(ret_shape);
  ASSERT_EQ(*ret_shape, *std::make_shared<abstract::TensorShape>(ShapeVector{16, 32}));
}

/// Feature: KernelPacket
/// Description: all device nodes (with dvm op) have only one user, and they are only shape depended.
/// Expectation: fuse all nodes in kernel packet
TEST_F(TestKernelPacket, only_depend_shape_2_basic_with_dvm) {
  auto fg = ConstructGraph_basic();
  SetGraphKernelFlags("--enable_cluster_ops_only=Add");
  SetDeviceTarget(kAscendDevice);
  RunPass(fg, {std::make_shared<StaticShapeCluster>(), std::make_shared<ConvertCallToPrim>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 1);
  RunPass(fg, {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  auto nodes = GetAllCNodes(fg);
  ASSERT_EQ(nodes.size(), 1);
  NodeShapeVector realshape = {
    {fg->parameters()[0], {16, 8}},   // p1
    {fg->parameters()[1], {8, 1}},    // w
    {fg->parameters()[2], {1, 32}},   // p2
    {fg->parameters()[3], {128, 4}},  // p3
  };
  ASSERT_TRUE(InferPacketNode(nodes[0], realshape));
  auto ret_shape = mindspore::symshape::QueryShape(nodes[0]->abstract());
  UT_CHECK_NULL(ret_shape);
  ASSERT_EQ(*ret_shape, *std::make_shared<abstract::TensorShape>(ShapeVector{16, 32}));
}

FuncGraphPtr ConstructGraph3() {
  ConstructGraph gb;
  auto p1 = gb.NewTensorInput("p1", kFloat32, {-1, -1});
  auto w = gb.NewTensorInput("w", kFloat32, {-1, -1});
  auto p2 = gb.NewTensorInput("p2", kFloat32, {-1, -1});
  auto trans = gb.NewValueNode(MakeValue<bool>(false));
  auto matmul = gb.NewCNodeWithBuildInfo("MatMul", {p1, w, trans, trans});
  auto add = gb.NewCNodeWithBuildInfo("Add", {matmul, p2});
  auto shape = gb.NewCNodeWithBuildInfo("Shape", {add});
  auto reshape = gb.NewCNodeWithBuildInfo("Reshape", {matmul, shape});
  gb.SetOutput(reshape);
  return gb.GetGraph();
}

/// Feature: KernelPacket
/// Description: the Add op has only one user and is only-shape-depended, the MatMul op has two users in packet node
/// Expectation: fuse Add in kernel packet
TEST_F(TestKernelPacket, only_depend_shape_3_multiuser) {
  auto fg = ConstructGraph3();
  RunPass(fg, {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  ASSERT_EQ(GetAllCNodes(fg).size(), 2);
  ASSERT_EQ(GetAllPacketNodes(fg).size(), 1);
}

FuncGraphPtr ConstructGraph4() {
  ConstructGraph gb;
  auto p1 = gb.NewTensorInput("p1", kFloat32, {-1, -1});
  auto w = gb.NewTensorInput("w", kFloat32, {-1, -1});
  auto p2 = gb.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = gb.NewTensorInput("p3", kFloat32, {-1, -1});
  auto trans = gb.NewValueNode(MakeValue<bool>(false));
  auto matmul = gb.NewCNodeWithBuildInfo("MatMul", {p1, w, trans, trans});
  auto add = gb.NewCNodeWithBuildInfo("Add", {matmul, p2});
  auto abs = gb.NewCNodeWithBuildInfo("Abs", {matmul});
  auto shape = gb.NewCNodeWithBuildInfo("Shape", {add});
  auto reshape = gb.NewCNodeWithBuildInfo("Reshape", {p3, shape});
  auto out = gb.NewCNode("MakeTuple", {abs, reshape});
  gb.SetOutput(out);
  return gb.GetGraph();
}

/// Feature: KernelPacket
/// Description: the Add op has only one user and is only-shape-depended, the MatMul op has two users, only one in
///              packet node.
/// Expectation: fuse Add in kernel packet
TEST_F(TestKernelPacket, only_depend_shape_4_multiuser) {
  auto fg = ConstructGraph4();
  RunPass(fg, {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  // MatMul, Abs, Reshape_packet, MakeTuple
  ASSERT_EQ(GetAllCNodes(fg).size(), 4);
  ASSERT_EQ(GetAllPacketNodes(fg).size(), 1);
}

FuncGraphPtr ConstructGraph5() {
  ConstructGraph gb;
  auto p1 = gb.NewTensorInput("p1", kFloat32, {-1, -1});
  auto p2 = gb.NewTensorInput("p2", kFloat32, {-1, -1});
  auto p3 = gb.NewTensorInput("p3", kFloat32, {-1, -1});
  auto p4 = gb.NewTensorInput("p4", kFloat32, {-1});
  auto axis = gb.NewCNodeWithBuildInfo("Shape", {p4});
  auto v_false = gb.NewValueNode(MakeValue<bool>(false));
  auto reduce = gb.NewCNodeWithBuildInfo("ReduceSum", {p3, axis, v_false, v_false});
  auto add = gb.NewCNodeWithBuildInfo("Add", {p1, reduce});
  auto shape = gb.NewCNodeWithBuildInfo("Shape", {add});
  auto reshape = gb.NewCNodeWithBuildInfo("Reshape", {p2, shape});
  gb.SetOutput(reshape);
  return gb.GetGraph();
}

/// Feature: KernelPacket
/// Description: the Add has only one user and can be fused, the ReduceSum also has only one user, when disable the
///               only-depend-shape pattern, it's fused into two packet nodes.
/// Expectation: make 2 kernelpacket node, and Add is not fused.
TEST_F(TestKernelPacket, only_depend_shape_5_two_packet) {
  auto fg = ConstructGraph5();
  MOCKER_CPP(packet::IsOnlyOneUser, bool (*)(const AnfNodePtr &)).stubs().will(returnValue(false));
  RunPass(fg, {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  ASSERT_EQ(GetAllCNodes(fg).size(), 3);
  ASSERT_EQ(GetAllPacketNodes(fg).size(), 2);
  GlobalMockObject::verify();
}

/// Feature: KernelPacket
/// Description: the Add has only one user and can be fused, the ReduceSum also has only one user, when enable the
///               only-depend-shape pattern, it's fused all nodes.
/// Expectation: fuse all nodes into one packet
TEST_F(TestKernelPacket, only_depend_shape_6_packet_in_packet) {
  auto fg = ConstructGraph5();
  RunPass(fg, {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  ASSERT_EQ(GetAllCNodes(fg).size(), 1);
}
}  // namespace mindspore::graphkernel::test
