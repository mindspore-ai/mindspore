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
#include "ir/graph_utils.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  std::vector<TypePtr> inputs_type;
};
}  // namespace

/// Feature: Test RsqrtGrad expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestRsqrtGradExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestRsqrtGradExpander, rsqrt_grad) {
  const auto &param = GetParam();
  ConstructGraph c;
  std::vector<AnfNodePtr> inputs(param.inputs_type.size());
  for (size_t i = 0; i < param.inputs_type.size(); ++i) {
    inputs[i] = c.NewTensorInput("x" + std::to_string(i), param.inputs_type[i], {10, 20});
  }
  auto op = c.NewCNodeWithBuildInfo("RsqrtGrad", inputs);
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpRsqrtGrad, TestRsqrtGradExpander,
                        testing::Values(Params{true, {kFloat16, kFloat16}}, Params{true, {kFloat32, kFloat32}},
                                        Params{true, {kBFloat16, kBFloat16}}, Params{false, {kFloat64, kFloat64}},
                                        Params{false, {kInt32, kInt32}}, Params{false, {kInt64, kInt64}},
                                        Params{false, {kInt8, kInt8}}));
}  // namespace mindspore::graphkernel::test
