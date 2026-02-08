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
struct RepeatInterleaveIntParams {
  bool can_expand;
  ShapeVector input_shape;
  int64_t repeat;
  int64_t dim;
  ShapeVector expect_shape;
  TypePtr type;
};
}  // namespace

/// Feature: Test graph kernel RepeatInterleaveInt expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestRepeatInterleaveIntExpander : public TestGraphKernelExpander,
                                        public testing::WithParamInterface<RepeatInterleaveIntParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestRepeatInterleaveIntExpander, RepeatInterleaveInt) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto shape = c.NewTensorInput("input_shape", param.type, param.input_shape);
  auto repeat = c.NewScalarInput("repeat", MakeValue(param.repeat), kInt64);
  auto dim = c.NewScalarInput("dim", MakeValue(param.dim), kInt64);
  auto output_size = c.NewValueNode(kNone);
  auto op = c.NewCNodeWithBuildInfo("RepeatInterleaveInt", {shape, repeat, dim, output_size}, {});
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

INSTANTIATE_TEST_CASE_P(TestOpRepeatInterleaveInt, TestRepeatInterleaveIntExpander,
                        testing::Values(RepeatInterleaveIntParams{true, {2, 3, 4, 5}, 2, 2, {2, 3, 8, 5}, kFloat16},
                                        RepeatInterleaveIntParams{true, {2, 3, 4, 5}, 2, -2, {2, 3, 8, 5}, kFloat32},
                                        RepeatInterleaveIntParams{true, {2, 3, 4, 5}, 2, 2, {2, 3, 8, 5}, kBFloat16},
                                        RepeatInterleaveIntParams{true, {2, 3, 4, 5}, 2, 2, {2, 3, 8, 5}, kInt32},
                                        RepeatInterleaveIntParams{false, {2, 3, 4, 5}, 2, 2, {2, 3, 8, 5}, kFloat64},
                                        RepeatInterleaveIntParams{false, {2, 3, 4, 5}, 2, 2, {2, 3, 8, 5}, kBool},
                                        RepeatInterleaveIntParams{false, {2, 3, 4, 5}, 2, 2, {2, 3, 8, 5}, kInt8},
                                        RepeatInterleaveIntParams{false, {2, 3, 4, 5}, 2, 2, {2, 3, 8, 5}, kInt64},
                                        RepeatInterleaveIntParams{false, {-2}, 2, 2, {-2}, kFloat32}));
}  // namespace mindspore::graphkernel::test