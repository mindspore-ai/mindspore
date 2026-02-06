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
struct LeakyReLUExtParams {
  bool can_expand;
  ShapeVector input_shape;
  ShapeVector expect_shape;
  std::vector<TypePtr> inputs_type;
};
}  // namespace

/// Feature: Test graph kernel LeakyReLUExt expander
/// Description: LeakyReLUExt will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestLeakyReLUExtExpander : public TestGraphKernelExpander,
                                 public testing::WithParamInterface<LeakyReLUExtParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestLeakyReLUExtExpander, LeakyReLUExt) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto shape = c.NewTensorInput("input_shape", param.inputs_type[0], param.input_shape);
  auto negative_slope = c.NewValueNode(NewScalar(param.inputs_type[1], 0.01, true));
  auto op = c.NewCNodeWithBuildInfo("LeakyReLUExt", {shape, negative_slope}, {});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, param.expect_shape, param.inputs_type[0]->type_id());
    }
  }
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  auto gknodes = GetAllGKNodes(g);
  size_t gk_size = param.can_expand ? 1 : 0;
  EXPECT_EQ(gknodes.size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpLeakyReLUExt, TestLeakyReLUExtExpander,
                        testing::Values(LeakyReLUExtParams{true, {16, 16}, {16, 16}, {kFloat16, kFloat32}},
                                        LeakyReLUExtParams{true, {16, 16}, {16, 16}, {kFloat32, kFloat32}},
                                        LeakyReLUExtParams{true, {16, 16}, {16, 16}, {kBFloat16, kFloat32}},
                                        LeakyReLUExtParams{false, {16, 16}, {16, 16}, {kFloat64, kFloat32}},
                                        LeakyReLUExtParams{false, {16, 16}, {16, 16}, {kFloat16, kFloat64}},
                                        LeakyReLUExtParams{false, {16, 16}, {16, 16}, {kFloat32, kFloat64}},
                                        LeakyReLUExtParams{false, {16, 16}, {16, 16}, {kBFloat16, kFloat64}},
                                        LeakyReLUExtParams{false, {16, 16}, {16, 16}, {kFloat64, kFloat64}}));
}  // namespace mindspore::graphkernel::test