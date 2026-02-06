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

#include <map>
#include <string>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "common/graph_optimizer_test_framework.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/convert_input_and_attr.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"
#include "ir/graph_utils.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  ShapeVector input_shape;
  ShapeVector expect_shape;
  TypePtr type;
  ShapeVector axis;
};

struct GradParams {
  bool can_expand;
  TypePtr type;
  ShapeVector shape;
  int64_t axis;
};
}  // namespace

/// Feature: Test graph kernel Softmax expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestSoftmaxExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    SetGraphKernelFlags("--enable_expand_ops=Softmax");
  }
};

TEST_P(TestSoftmaxExpander, softmax) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x = c.NewTensorInput("x", param.type, param.input_shape);
  auto axis = c.NewValueNode(MakeValue(param.axis));
  auto op = c.NewCNodeWithBuildInfo("Softmax", {x, axis});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
                         std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
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

/// Feature: Test graph kernel SoftmaxBackward expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestSoftmaxBackwardExpander : public TestGraphKernelExpander, public testing::WithParamInterface<GradParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    SetGraphKernelFlags("--enable_expand_ops=SoftmaxBackward");
  }
};

TEST_P(TestSoftmaxBackwardExpander, softmax_grad) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto dout = c.NewTensorInput("dout", param.type, param.shape);
  auto out = c.NewTensorInput("out", param.type, param.shape);
  auto axis = c.NewValueNode(MakeValue(param.axis));
  auto op = c.NewCNodeWithBuildInfo("SoftmaxBackward", {dout, out, axis});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(c.GetGraph()).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpSoftmax, TestSoftmaxExpander,
                        testing::Values(Params{true, {16, 16}, {16, 16}, kFloat32, {-1}},
                                        Params{true, {16, 16}, {16, 16}, kFloat16, {-1}},
                                        Params{true, {16, 16}, {16, 16}, kBFloat16, {-1}}));

INSTANTIATE_TEST_CASE_P(TestOpSoftmaxBackward, TestSoftmaxBackwardExpander,
                        testing::Values(GradParams{true, kFloat32, {16, 16}, 1},
                                        GradParams{true, kFloat16, {16, 16}, 1},
                                        GradParams{true, kBFloat16, {16, 16}, 1},
                                        GradParams{false, kBFloat16, {-2}, -1}, GradParams{false, kBFloat16, {-2}, 1},
                                        GradParams{false, kBFloat16, {16, 16}, 0},
                                        GradParams{false, kBFloat16, {16, 16}, -2}));
}  // namespace mindspore::graphkernel::test