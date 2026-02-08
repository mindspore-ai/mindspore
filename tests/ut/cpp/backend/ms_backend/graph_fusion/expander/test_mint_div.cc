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
#include "op_enum.h"
#include "ir/graph_utils.h"

namespace mindspore::graphkernel::test {
namespace {
struct DivModParams {
  bool can_expand;
  ShapeVector x1_shape;
  ShapeVector x2_shape;
  ops::RoundingMode rounding_mode;
  ShapeVector expect_shape;
  TypePtr type;
  bool is_none{false};
};
}  // namespace

/// Feature: Test graph kernel DivMod expander
/// Description: DivMod will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestDivModExpander : public TestGraphKernelExpander, public testing::WithParamInterface<DivModParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestDivModExpander, DivMod) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x1 = c.NewTensorInput("x1_shape", param.type, param.x1_shape);
  auto x2 = c.NewTensorInput("x2_shape", param.type, param.x2_shape);
  AnfNodePtr rounding_mode;
  if (param.is_none) {
    rounding_mode = c.NewValueNode(kNone);
  } else {
    rounding_mode = c.NewValueNode(NewScalar(kInt64, static_cast<int64_t>(param.rounding_mode), true));
  }
  auto op = c.NewCNodeWithBuildInfo("DivMod", {x1, x2, rounding_mode}, {});
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

INSTANTIATE_TEST_CASE_P(
  TestOpDivMod, TestDivModExpander,
  testing::Values(DivModParams{true, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kFloat16, true},
                  DivModParams{true, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kFloat16},
                  DivModParams{true, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kFloat32},
                  DivModParams{true, {16, 16}, {16, 16}, ops::RoundingMode::TRUNC, {16, 16}, kFloat16},
                  DivModParams{true, {16, 16}, {16, 16}, ops::RoundingMode::TRUNC, {16, 16}, kFloat32},
                  DivModParams{true, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kBFloat16},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::ROUND, {16, 16}, kFloat16},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::CEIL, {16, 16}, kFloat16},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kFloat64},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kInt64},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kInt32},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kInt16},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kInt8},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kUInt8},
                  DivModParams{false, {16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kBool}));
}  // namespace mindspore::graphkernel::test