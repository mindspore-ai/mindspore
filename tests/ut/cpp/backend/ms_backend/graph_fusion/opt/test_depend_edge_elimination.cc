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

#include <map>
#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/depend_edge_elimination.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "primitive/sequence_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore::graphkernel::test {
/// Feature: Test depend_edge_elimination pass
/// Description: depend_edge_elimination pass will skip output node
/// Expectation: After depend_edge_elimination pass, the output node should still be Assign.
TEST_F(GraphKernelCommonTestSuite, depend_edge_elimination_output_node) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops=MatMul");
  ConstructGraph gb;
  test::ConstructGraph c;
  auto p0 = c.NewTensorInput("p0", kFloat16, {1024, 1024});
  auto x1 = c.NewTensorInput("x1", kFloat16, {1024, 1024});
  auto x2 = c.NewTensorInput("x2", kFloat16, {1024, 1024});
  auto x3 = c.NewTensorInput("x3", kFloat16, {1024, 1024});

  auto x1_ext = c.NewCNodeWithBuildInfo("Cos", {x1}, {});
  auto x2_ext = c.NewCNodeWithBuildInfo("Sin", {x2}, {});

  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));

  auto op0 = c.NewCNodeWithBuildInfo("MatMul", {x1, x2, trans_a, trans_b}, {});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {op0, x3}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Assign", {p0, op1}, {});

  auto op3 = c.NewCNodeWithBuildInfo("Depend", {x1_ext, op2}, {});
  auto op4 = c.NewCNodeWithBuildInfo("Depend", {x2_ext, op0}, {});
  auto op5 = c.NewCNodeWithBuildInfo("MakeTuple", {op3, op4}, {});
  c.SetOutput(op5);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<ConvertFrontEndToGraphKernel>(), std::make_shared<graphkernel::StaticShapeCluster>(),
               std::make_shared<DependEdgeElimination>()});
  auto gk_nodes = GetAllGKNodes(fg);
  EXPECT_EQ(gk_nodes.size(), 1);
  auto graph = GetCNodeFuncGraph(gk_nodes[0]);
  EXPECT_TRUE(IsPrimitiveCNode(graph->output(), prim::kPrimAssign));
}

}  // namespace mindspore::graphkernel::test
