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

#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_cluster_cloud.h"
#include "backend/ms_backend/graph_fusion/add_atomic_clean.h"
#include "backend/ms_backend/graph_fusion/insert_pad.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  ShapeArray input_shape;
  bool input0_pad;
  bool input1_pad;
  bool unpad;
};
}  // namespace

/// Feature: Test graph kernel InsertPadOps pass
/// Description: InsertPadOps pass
/// Expectation: After pass, PadAkg will be inserted
class TestInsertPadOps : public GraphKernelCommonTestSuite, public testing::WithParamInterface<Params> {};

TEST_P(TestInsertPadOps, insert_pad_ops) {
  const auto &param = GetParam();
  SetDeviceTarget(kGPUDevice);
  SetGraphKernelFlags("--enable_cluster_ops=MatMul");
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat16, param.input_shape[0]);
  auto x1 = c.NewTensorInput("x1", kFloat16, param.input_shape[1]);
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(false));
  auto y0 = c.NewCNodeWithBuildInfo("MatMul", {x0, x1, trans_a, trans_b});
  c.SetOutput(y0);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
               std::make_shared<graphkernel::StaticShapeCluster>(), std::make_shared<graphkernel::InsertPadOps>()});
  bool check = false;
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto output = sub_graph->get_return()->input(1);
      auto prim = common::AnfAlgo::GetCNodePrimitive(output);
      MS_EXCEPTION_IF_NULL(prim);
      auto prim_name = prim->name();
      std::string target_name = param.unpad ? "UnPadAkg" : "MatMul";
      EXPECT_EQ(prim_name, target_name);
      auto matmul_node = param.unpad ? output->cast<CNodePtr>()->input(1) : output;
      auto input0 = matmul_node->cast<CNodePtr>()->input(1);
      auto input1 = matmul_node->cast<CNodePtr>()->input(2);
      EXPECT_EQ((input0 != nullptr && input1 != nullptr), true);
      EXPECT_EQ(input0->isa<CNode>(), param.input0_pad);
      EXPECT_EQ(input1->isa<CNode>(), param.input1_pad);
      check = true;
      break;
    }
  }
  EXPECT_EQ(check, true);
}

INSTANTIATE_TEST_CASE_P(TestPassInsertPadOps, TestInsertPadOps,
                        testing::Values(Params{{{1023, 515}, {515, 2044}}, true, true, true},
                                        Params{{{1024, 512}, {512, 2048}}, false, false, false}));
}  // namespace mindspore::graphkernel::test
