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
#include "backend/ms_backend/graph_fusion/tensor_inplace.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore::graphkernel::test {
class TestPassTensorInplace : public GraphKernelCommonTestSuite {};

/// Feature: Test graph kernel TensorInplace pass
/// Description: TensorInplace pass
/// Expectation: After pass, Assign node will be inserted to reuse sub graph's input device address
TEST_F(TestPassTensorInplace, tensor_inplace) {
  SetDeviceTarget(kAscendDevice);
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {4, 16});
  auto x1 = c.NewTensorInput("x1", kFloat32, {16, 4});
  auto x2 = c.NewTensorInput("x3", kFloat32, {4, 4});
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto y0 = c.NewCNodeWithBuildInfo("MatMul", {x0, x1, trans_a, trans_b});
  auto y1 = c.NewCNodeWithBuildInfo("Sub", {y0, x2});
  auto y2 = c.NewCNodeWithBuildInfo("Exp", {y1});
  auto y3 = c.NewCNodeWithBuildInfo("Add", {y2, x2});
  c.SetOutput(y3);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<TensorInplace>()});
  bool check = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto output = sub_graph->get_return()->input(1);
      check = IsPrimitiveCNode(output, prim::kPrimAssign);
      break;
    }
  }
  EXPECT_EQ(check, true);
}
}  // namespace mindspore::graphkernel::test
