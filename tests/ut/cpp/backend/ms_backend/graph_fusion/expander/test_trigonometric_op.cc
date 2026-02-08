/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include <iostream>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"
#include "ir/graph_utils.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  std::string op_name;
  ShapeVector input_shape;
  ShapeVector expect_shape;
  TypePtr type;
};

struct GradParams {
  bool can_expand;
  TypePtr type;
  ShapeVector shape;
};
}  // namespace

/// Feature: Test graph kernel Tanh/Cosh/... expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestTrigonometricExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestTrigonometricExpander, trigonometric_op) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x = c.NewTensorInput("x", param.type, param.input_shape);
  auto op = c.NewCNodeWithBuildInfo(param.op_name, {x});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, param.expect_shape, param.type->type_id());
    }
  }
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  auto gknodes = GetAllGKNodes(g);
  size_t gk_size = param.can_expand ? 1 : 0;
  EXPECT_EQ(gknodes.size(), gk_size);
}

/// Feature: Test graph kernel TanhGrad expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestTanhGradExpander : public TestGraphKernelExpander, public testing::WithParamInterface<GradParams> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestTanhGradExpander, tanh_grad) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto y = c.NewTensorInput("y", param.type, param.shape);
  auto dy = c.NewTensorInput("dy", param.type, param.shape);
  auto op = c.NewCNodeWithBuildInfo("TanhGrad", {y, dy});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(c.GetGraph()).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(
  TestOpTrigonometric, TestTrigonometricExpander,
  testing::Values(
    Params{true, "Tanh", {16, 16}, {16, 16}, kFloat32}, Params{false, "Tanh", {16, 16}, {16, 16}, kFloat16},
    Params{false, "Tanh", {16, 16}, {16, 16}, kBFloat16}, Params{false, "Tanh", {16, 16}, {16, 16}, kFloat64},
    Params{true, "Cosh", {16, 16}, {16, 16}, kFloat32}, Params{false, "Cosh", {16, 16}, {16, 16}, kFloat16},
    Params{false, "Cosh", {16, 16}, {16, 16}, kBFloat16}, Params{false, "Cosh", {16, 16}, {16, 16}, kFloat64},
    Params{true, "Sinh", {16, 16}, {16, 16}, kFloat32}, Params{false, "Sinh", {16, 16}, {16, 16}, kFloat16},
    Params{false, "Sinh", {16, 16}, {16, 16}, kBFloat16}, Params{false, "Sinh", {16, 16}, {16, 16}, kFloat64},
    Params{true, "AcoshExt", {16, 16}, {16, 16}, kFloat32}, Params{true, "AcoshExt", {16, 16}, {16, 16}, kFloat16},
    Params{true, "AcoshExt", {16, 16}, {16, 16}, kBFloat16}, Params{false, "AcoshExt", {16, 16}, {16, 16}, kFloat64},
    Params{true, "AsinhExt", {16, 16}, {16, 16}, kFloat32}, Params{true, "AsinhExt", {16, 16}, {16, 16}, kFloat16},
    Params{true, "AsinhExt", {16, 16}, {16, 16}, kBFloat16}, Params{false, "AsinhExt", {16, 16}, {16, 16}, kFloat64}));

INSTANTIATE_TEST_CASE_P(TestOpTanhGrad, TestTanhGradExpander,
                        testing::Values(GradParams{true, kFloat16, {16, 16}}, GradParams{true, kFloat32, {16, 16}},
                                        GradParams{true, kBFloat16, {16, 16}}, GradParams{true, kBFloat16, {-1, -1}},
                                        GradParams{false, kFloat64, {16, 16}}, GradParams{false, kInt64, {16, 16}}));
}  // namespace mindspore::graphkernel::test