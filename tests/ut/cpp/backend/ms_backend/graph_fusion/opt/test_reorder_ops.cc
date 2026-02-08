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

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/reorder_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "utils/anf_utils.h"
#include "include/utils/anfalgo.h"

namespace mindspore::graphkernel::test {
class TestPassReorderOps : public GraphKernelCommonTestSuite {};

/// Feature: Test graph kernel ReorderOps pass
/// Description: Cast up case
/// Expectation: Reorder cast node and type insensitive node
TEST_F(TestPassReorderOps, cast_up) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops=Transpose");
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat16, {4, 16});
  auto y0 = c.NewCNodeWithBuildInfo("Cast", {x0, c.NewValueNode<int64_t>(kNumberTypeFloat32)});
  auto y1 = c.NewCNodeWithBuildInfo("Transpose", {y0, c.NewValueNode(MakeValue(ShapeVector{1, 0}))});
  auto y2 = c.NewCNodeWithBuildInfo("Neg", {y1});
  c.SetOutput(y2);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<ReorderOps>()});
  bool check = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto cast_node = sub_graph->get_return()->input(1);
      EXPECT_EQ(IsPrimitiveCNode(cast_node, prim::kPrimCast), true);
      CheckInputOutputType(cast_node, {kNumberTypeFloat16}, kNumberTypeFloat32);
      auto neg_node = cast_node->cast<CNodePtr>()->input(1);
      EXPECT_EQ(IsPrimitiveCNode(neg_node, prim::kPrimNeg), true);
      CheckInputOutputType(neg_node, {kNumberTypeFloat16}, kNumberTypeFloat16);
      auto transpose_node = neg_node->cast<CNodePtr>()->input(1);
      EXPECT_EQ(IsPrimitiveCNode(transpose_node, prim::kPrimTranspose), true);
      CheckInputOutputType(transpose_node, {kNumberTypeFloat16}, kNumberTypeFloat16);
      check = true;
      break;
    }
  }
  EXPECT_EQ(check, true);
}

/// Feature: Test graph kernel ReorderOps pass
/// Description: Cast down case
/// Expectation: Reorder cast node and type insensitive node
TEST_F(TestPassReorderOps, cast_down) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops=Transpose");
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {4, 16});
  auto y0 = c.NewCNodeWithBuildInfo("Transpose", {x0, c.NewValueNode(MakeValue(ShapeVector{1, 0}))});
  auto y1 = c.NewCNodeWithBuildInfo("Neg", {y0});
  auto y2 = c.NewCNodeWithBuildInfo("Cast", {y1, c.NewValueNode<int64_t>(kNumberTypeFloat16)});
  c.SetOutput(y2);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<ReorderOps>()});
  bool check = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto neg_node = sub_graph->get_return()->input(1);
      EXPECT_EQ(IsPrimitiveCNode(neg_node, prim::kPrimNeg), true);
      CheckInputOutputType(neg_node, {kNumberTypeFloat16}, kNumberTypeFloat16);
      auto transpose_node = neg_node->cast<CNodePtr>()->input(1);
      EXPECT_EQ(IsPrimitiveCNode(transpose_node, prim::kPrimTranspose), true);
      CheckInputOutputType(transpose_node, {kNumberTypeFloat16}, kNumberTypeFloat16);
      auto cast_node = transpose_node->cast<CNodePtr>()->input(1);
      EXPECT_EQ(IsPrimitiveCNode(cast_node, prim::kPrimCast), true);
      CheckInputOutputType(cast_node, {kNumberTypeFloat32}, kNumberTypeFloat16);
      check = true;
      break;
    }
  }
  EXPECT_EQ(check, true);
}
}  // namespace mindspore::graphkernel::test
