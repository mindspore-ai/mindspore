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
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "ir/graph_utils.h"

namespace mindspore::graphkernel::test {
namespace {
struct SigmoidParams {
  bool can_expand;
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

/// Feature: Test graph kernel Sigmoid expander
/// Description: Sigmoid will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestSigmoidExpander : public TestGraphKernelExpander, public testing::WithParamInterface<SigmoidParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestSigmoidExpander, Sigmoid) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto shape = c.NewTensorInput("input_shape", param.type, param.input_shape);
  auto op = c.NewCNodeWithBuildInfo("Sigmoid", {shape}, {});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      auto fg = GetCNodeFuncGraph(node);
      auto output_node = fg->output();
      if (IsPrimitiveCNode(output_node, prim::kPrimCast)) {
        auto cnode = output_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        output_node = cnode->input(1);
      }
      CompareShapeAndType(output_node, 0, param.expect_shape, kFloat32->type_id());
    }
  }
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(c.GetGraph()).size(), gk_size);
}

/// Feature: Test graph kernel SigmoidGrad expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestSigmoidGradExpander : public TestGraphKernelExpander, public testing::WithParamInterface<GradParams> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestSigmoidGradExpander, sigmoid_grad) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto y = c.NewTensorInput("y", param.type, param.shape);
  auto dy = c.NewTensorInput("dy", param.type, param.shape);
  auto op = c.NewCNodeWithBuildInfo("SigmoidGrad", {y, dy});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(c.GetGraph()).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpSigmoid, TestSigmoidExpander,
                        testing::Values(SigmoidParams{true, {16, 16}, {16, 16}, kFloat16},
                                        SigmoidParams{true, {16, 16}, {16, 16}, kFloat32},
                                        SigmoidParams{true, {16, 16}, {16, 16}, kBFloat16},
                                        SigmoidParams{false, {16, 16}, {16, 16}, kInt64}));

INSTANTIATE_TEST_CASE_P(TestOpSigmoidGrad, TestSigmoidGradExpander,
                        testing::Values(GradParams{true, kFloat16, {16, 16}}, GradParams{true, kFloat32, {16, 16}},
                                        GradParams{true, kBFloat16, {16, 16}}, GradParams{true, kBFloat16, {-1, -1}},
                                        GradParams{false, kInt64, {16, 16}}));
}  // namespace mindspore::graphkernel::test