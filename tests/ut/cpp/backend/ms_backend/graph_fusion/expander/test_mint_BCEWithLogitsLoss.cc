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
struct BCEWithLogitsLossParams {
  ShapeVector logits;
  ShapeVector labels;
  ShapeVector weight;
  ShapeVector pos_weight;
  int64_t mode;  // 0-----None, 1------reducemean, 2-------reducesum
  ShapeVector expect_shape;
  TypePtr type;
};
}  // namespace

/// Feature: Test graph kernel BCEWithLogitsLoss expander
/// Description: BCEWithLogitsLoss will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestBCEWithLogitsLossExpander : public TestGraphKernelExpander,
                                      public testing::WithParamInterface<BCEWithLogitsLossParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM --enable_expand_ops=BCEWithLogitsLoss";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestBCEWithLogitsLossExpander, BCEWithLogitsLoss) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto logits = c.NewTensorInput("logits", param.type, param.logits);
  auto labels = c.NewTensorInput("labels", param.type, param.labels);
  auto weight = c.NewTensorInput("weight", param.type, param.weight);
  auto pos_weight = c.NewTensorInput("pos_weight", param.type, param.pos_weight);
  auto mode = c.NewScalarInput("mode", MakeValue(param.mode), kInt64);
  auto op = c.NewCNodeWithBuildInfo("BCEWithLogitsLoss", {logits, labels, weight, pos_weight, mode}, {});
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
  EXPECT_EQ(gknodes.size(), 1);
}

INSTANTIATE_TEST_CASE_P(
  TestOpBCEWithLogitsLoss, TestBCEWithLogitsLossExpander,
  testing::Values(BCEWithLogitsLossParams{{16, 16}, {16, 16}, {16, 16}, {16, 16}, 0, {}, kFloat16},
                  BCEWithLogitsLossParams{{16, 16}, {16, 16}, {16, 16}, {16, 16}, 0, {}, kFloat32},
                  BCEWithLogitsLossParams{{16, 16}, {16, 16}, {16, 16}, {16, 16}, 1, {}, kFloat16},
                  BCEWithLogitsLossParams{{16, 16}, {16, 16}, {16, 16}, {16, 16}, 1, {}, kFloat32},
                  BCEWithLogitsLossParams{{16, 16}, {16, 16}, {16, 16}, {16, 16}, 2, {16, 16}, kFloat16},
                  BCEWithLogitsLossParams{{16, 16}, {16, 16}, {16, 16}, {16, 16}, 2, {16, 16}, kFloat32}));
}  // namespace mindspore::graphkernel::test