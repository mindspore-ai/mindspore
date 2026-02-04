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
  std::string op_name;
  std::vector<TypePtr> inputs_type;
};
}  // namespace

/// Feature: Test GeLU/... activation op expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestActivationExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestActivationExpander, activation_op) {
  const auto &param = GetParam();
  ConstructGraph c;
  std::vector<AnfNodePtr> inputs(param.inputs_type.size());
  for (size_t i = 0; i < param.inputs_type.size(); ++i) {
    inputs[i] = c.NewTensorInput("x" + std::to_string(i), param.inputs_type[i], {10, 20});
  }
  auto op = c.NewCNodeWithBuildInfo(param.op_name, inputs);
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(
  TestOpActivation, TestActivationExpander,
  testing::Values(
    // GeLU
    Params{true, "GeLU", {kFloat16}}, Params{true, "GeLU", {kFloat32}}, Params{true, "GeLU", {kBFloat16}},
    Params{false, "GeLU", {kFloat64}},
    // GeLUGrad
    Params{true, "GeLUGrad", {kFloat16, kFloat16}}, Params{true, "GeLUGrad", {kFloat32, kFloat32}},
    Params{true, "GeLUGrad", {kBFloat16, kBFloat16}}, Params{false, "GeLUGrad", {kInt32, kInt32}},
    Params{false, "GeLUGrad", {kInt64, kInt64}}, Params{false, "GeLUGrad", {kInt8, kInt8}},
    // FastGeLU
    Params{true, "FastGeLU", {kFloat16}}, Params{true, "FastGeLU", {kFloat32}}, Params{true, "FastGeLU", {kBFloat16}},
    // FastGeLUGrad
    Params{true, "FastGeLUGrad", {kFloat16, kFloat16}}, Params{true, "FastGeLUGrad", {kFloat32, kFloat32}},
    Params{true, "FastGeLUGrad", {kBFloat16, kBFloat16}},
    // ReLU
    Params{true, "ReLU", {kFloat16}}, Params{true, "ReLU", {kFloat32}}, Params{true, "ReLU", {kBFloat16}},
    Params{true, "ReLU", {kInt32}}, Params{false, "ReLU", {kInt64}}, Params{false, "ReLU", {kInt8}},
    // ReluGrad
    Params{true, "ReluGrad", {kFloat16, kFloat16}}, Params{true, "ReluGrad", {kFloat32, kFloat32}},
    Params{true, "ReluGrad", {kBFloat16, kBFloat16}}, Params{false, "ReluGrad", {kInt32, kInt32}},
    Params{false, "ReluGrad", {kInt64, kInt64}}, Params{false, "ReluGrad", {kInt8, kInt8}},
    // SiLU
    Params{true, "SiLU", {kFloat16}}, Params{true, "SiLU", {kFloat32}}, Params{true, "SiLU", {kBFloat16}},
    Params{false, "SiLU", {kInt32}}, Params{false, "SiLU", {kInt64}}, Params{false, "SiLU", {kInt8}},
    // SiLUGrad
    Params{true, "SiLUGrad", {kFloat16, kFloat16}}, Params{true, "SiLUGrad", {kFloat32, kFloat32}},
    Params{true, "SiLUGrad", {kBFloat16, kBFloat16}}, Params{false, "SiLUGrad", {kInt32, kInt32}},
    Params{false, "SiLUGrad", {kInt64, kInt64}}, Params{false, "SiLUGrad", {kInt8, kInt8}}));
}  // namespace mindspore::graphkernel::test
