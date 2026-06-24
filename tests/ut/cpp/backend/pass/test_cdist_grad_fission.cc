/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <memory>
#include <string>
#include <vector>

#include "backend/common/pass/ir_fission/cdist_grad_fission.h"
#include "common/common_test.h"
#include "common/graph_optimizer_test_framework.h"
#include "include/utils/anfalgo.h"
#include "ir/graph_utils.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ops/test_value_utils.h"

namespace mindspore {
namespace {
ShapeVector GetNodeShape(const AnfNodePtr &node) { return common::AnfAlgo::GetOutputInferShape(node, 0); }

int64_t GetAxisAttr(const CNodePtr &node) { return common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrAxis); }

int64_t GetBatchRankAttr(const CNodePtr &node) { return common::AnfAlgo::GetNodeAttr<int64_t>(node, ops::kBatchRank); }

size_t CountPrimitive(const FuncGraphPtr &graph, const std::string &prim_name) {
  size_t count = 0;
  for (const auto &node : TopoSort(graph->get_return())) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto prim = GetCNodePrimitive(cnode);
    if (prim != nullptr && prim->name() == prim_name) {
      ++count;
    }
  }
  return count;
}
}  // namespace

class CdistGradFissionPassTest : public UT::Common {
 public:
  CdistGradFissionPassTest() = default;
  ~CdistGradFissionPassTest() override = default;
};

/// Feature: backend CdistGrad fission pass
/// Description: Rewrite the remaining rank-3 square-output CdistGrad path into broadcasted inputs.
/// Expectation: The pass inserts four ExpandDims and four BroadcastTo nodes, and rewires CdistGrad inputs.
TEST_F(CdistGradFissionPassTest, test_cdist_grad_fission_square_output_rewrite) {
  test::ConstructGraph c;
  auto grad = c.NewTensorInput("grad", kFloat32, {4, 8, 8});
  auto x = c.NewTensorInput("x", kFloat32, {4, 8, 16});
  auto y = c.NewTensorInput("y", kFloat32, {4, 8, 16});
  auto out = c.NewTensorInput("out", kFloat32, {4, 8, 8});
  auto p = c.NewScalarInput("p", ops::CreateScalar<float>(2.0f), kFloat32);
  auto cdist_grad = c.NewCNode("CdistGrad", {grad, x, y, out, p});
  c.SetOutput(cdist_grad);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::CdistGradFission>()});

  auto output = c.GetGraph()->output()->cast<CNodePtr>();
  ASSERT_NE(output, nullptr);
  EXPECT_NE(output, cdist_grad);
  ASSERT_NE(GetCNodePrimitive(output), nullptr);
  EXPECT_EQ(GetCNodePrimitive(output)->name(), "CdistGrad");
  EXPECT_EQ(GetBatchRankAttr(output), 1);
  EXPECT_EQ(CountPrimitive(c.GetGraph(), "ExpandDims"), 4);
  EXPECT_EQ(CountPrimitive(c.GetGraph(), "BroadcastTo"), 4);

  const std::vector<ShapeVector> expected_expand_shapes = {
    {4, 8, 8, 1},
    {4, 8, 1, 16},
    {4, 1, 8, 16},
    {4, 8, 8, 1},
  };
  const std::vector<int64_t> expected_axes = {-1, -2, -3, -1};
  const ShapeVector expected_broadcast_shape = {4, 8, 8, 16};
  for (size_t i = 1; i <= 4; ++i) {
    auto broadcast = output->input(i)->cast<CNodePtr>();
    ASSERT_NE(broadcast, nullptr);
    ASSERT_NE(GetCNodePrimitive(broadcast), nullptr);
    EXPECT_EQ(GetCNodePrimitive(broadcast)->name(), "BroadcastTo");
    EXPECT_EQ(GetNodeShape(broadcast), expected_broadcast_shape);
    auto expand_dims = broadcast->input(1)->cast<CNodePtr>();
    ASSERT_NE(expand_dims, nullptr);
    ASSERT_NE(GetCNodePrimitive(expand_dims), nullptr);
    EXPECT_EQ(GetCNodePrimitive(expand_dims)->name(), "ExpandDims");
    EXPECT_EQ(GetNodeShape(expand_dims), expected_expand_shapes[i - 1]);
    EXPECT_EQ(GetAxisAttr(expand_dims), expected_axes[i - 1]);
  }
  EXPECT_EQ(output->input(5), p);
}

/// Feature: backend CdistGrad fission pass
/// Description: Batch-broadcast cases are already handled by frontend fallback.
/// Expectation: The pass keeps the original CdistGrad node unchanged.
TEST_F(CdistGradFissionPassTest, test_cdist_grad_fission_skip_batch_broadcast_case) {
  test::ConstructGraph c;
  auto grad = c.NewTensorInput("grad", kFloat32, {5, 5, 3});
  auto x = c.NewTensorInput("x", kFloat32, {1, 5, 20});
  auto y = c.NewTensorInput("y", kFloat32, {5, 3, 20});
  auto out = c.NewTensorInput("out", kFloat32, {5, 5, 3});
  auto p = c.NewScalarInput("p", ops::CreateScalar<float>(2.0f), kFloat32);
  auto cdist_grad = c.NewCNode("CdistGrad", {grad, x, y, out, p});
  c.SetOutput(cdist_grad);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::CdistGradFission>()});

  EXPECT_EQ(c.GetGraph()->output(), cdist_grad);
  EXPECT_EQ(CountPrimitive(c.GetGraph(), "ExpandDims"), 0);
  EXPECT_EQ(CountPrimitive(c.GetGraph(), "BroadcastTo"), 0);
}

/// Feature: backend CdistGrad fission pass
/// Description: Non-square rank-3 cases are already handled by frontend fallback on Ascend.
/// Expectation: The pass keeps the original CdistGrad node unchanged.
TEST_F(CdistGradFissionPassTest, test_cdist_grad_fission_skip_non_square_case) {
  test::ConstructGraph c;
  auto grad = c.NewTensorInput("grad", kFloat32, {4, 7, 8});
  auto x = c.NewTensorInput("x", kFloat32, {4, 7, 16});
  auto y = c.NewTensorInput("y", kFloat32, {4, 8, 16});
  auto out = c.NewTensorInput("out", kFloat32, {4, 7, 8});
  auto p = c.NewScalarInput("p", ops::CreateScalar<float>(2.0f), kFloat32);
  auto cdist_grad = c.NewCNode("CdistGrad", {grad, x, y, out, p});
  c.SetOutput(cdist_grad);
  c.GetGraph()->set_run_mode(device::RunMode::kKernelMode);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::CdistGradFission>()});

  EXPECT_EQ(c.GetGraph()->output(), cdist_grad);
  EXPECT_EQ(CountPrimitive(c.GetGraph(), "ExpandDims"), 0);
  EXPECT_EQ(CountPrimitive(c.GetGraph(), "BroadcastTo"), 0);
}
}  // namespace mindspore
