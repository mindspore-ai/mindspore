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
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/optimize_assign.h"
#include "primitive/sequence_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::graphkernel::test {
namespace {
FuncGraphPtr ConstructGraph_1() {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat32, {3, 2});
  auto x1 = c.NewTensorInput("x1", kFloat32, {3, 2});
  auto x2 = c.NewTensorInput("x2", kFloat32, {3, 2});
  auto op0 = c.NewCNodeWithBuildInfo("Sub", {x1, x2}, {});
  auto op1 = c.NewCNodeWithBuildInfo("Assign", {p0, op0}, {});
  auto u = c.NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto op2 = c.NewCNodeWithBuildInfo("UpdateState", {u, op1}, {});
  auto op3 = c.NewCNodeWithBuildInfo("MakeTuple", {op0, op2}, {});
  c.SetOutput(op3);
  return c.GetGraph();
}
}  // namespace

/// Feature: Test optimize assign pass
/// Description: optimize assign pass will skip output node
/// Expectation: After optimize assign pass, the output node should still be Sub.
TEST_F(GraphKernelCommonTestSuite, optimize_assign_skip_output_node) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops=Sub,Assign");
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::OptimizeAssign>()});
  EXPECT_EQ(GetAllGKNodes(fg).size(), 1);
  auto output = fg->output();
  EXPECT_TRUE(IsPrimitiveCNode(fg->output(), prim::kPrimMakeTuple));
  auto node1 = output->cast<CNodePtr>()->input(1);
  EXPECT_TRUE(IsPrimitiveCNode(node1, prim::kPrimTupleGetItem));
  auto itemidx_node = node1->cast<CNodePtr>()->input(2);
  auto idx = GetValue<int64_t>(GetValueNode(itemidx_node));
  EXPECT_EQ(idx, 0);
}

}  // namespace mindspore::graphkernel::test
