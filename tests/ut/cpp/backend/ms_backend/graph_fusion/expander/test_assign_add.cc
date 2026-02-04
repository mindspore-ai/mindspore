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
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  std::vector<TypePtr> inputs_type;
  ShapeArray inputs_shape;
};
}  // namespace

/// Feature: Test AssignAdd expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestAssignAddExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestAssignAddExpander, assign_add) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", param.inputs_type[0], param.inputs_shape[0]);
  auto x1 = c.NewTensorInput("x1", param.inputs_type[1], param.inputs_shape[1]);
  auto u = c.NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto op = c.NewCNodeWithBuildInfo("AssignAdd", {x0, x1, u});
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(
  TestOpAssignAdd, TestAssignAddExpander,
  testing::Values(
    Params{true, {kFloat16, kFloat16}, {{2, 3}, {2, 3}}}, Params{true, {kFloat32, kFloat32}, {{2, 3}, {2, 3}}},
    Params{true, {kBFloat16, kBFloat16}, {{2, 3}, {2, 3}}}, Params{true, {kInt32, kInt32}, {{2, 3}, {2, 3}}},
    Params{true, {kFloat16, kFloat32}, {{2, 3}, {2, 3}}}, Params{true, {kFloat32, kFloat16}, {{2, 3}, {2, 3}}},
    Params{true, {kFloat32, kBFloat16}, {{2, 3}, {2, 3}}}, Params{true, {kBFloat16, kFloat16}, {{2, 3}, {2, 3}}},
    Params{true, {kBFloat16, kFloat32}, {{2, 3}, {2, 3}}}, Params{false, {kFloat64, kFloat64}, {{2, 3}, {2, 3}}},
    Params{false, {kInt8, kInt8}, {{2, 3}, {2, 3}}}, Params{false, {kInt16, kInt16}, {{2, 3}, {2, 3}}},
    Params{false, {kInt64, kInt64}, {{2, 3}, {2, 3}}}));
}  // namespace mindspore::graphkernel::test
