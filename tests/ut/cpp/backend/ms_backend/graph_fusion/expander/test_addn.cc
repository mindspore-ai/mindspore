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
  TypePtr input_type;
  size_t input_num;
};
}  // namespace

/// Feature: Test AddN expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestAddNExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestAddNExpander, addn) {
  const auto &param = GetParam();
  ConstructGraph c;
  std::vector<AnfNodePtr> inputs(param.input_num);
  for (size_t i = 0; i < param.input_num; ++i) {
    inputs[i] = c.NewTensorInput("x" + std::to_string(i), param.input_type, {10, 20});
  }
  auto op = c.NewCNodeWithBuildInfo("AddN", inputs);
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpAddN, TestAddNExpander,
                        testing::Values(Params{true, kFloat16, 3}, Params{true, kFloat32, 10},
                                        Params{true, kBFloat16, 3}, Params{true, kInt32, 4}, Params{false, kInt64, 3},
                                        Params{false, kInt16, 3}, Params{false, kInt8, 3}, Params{false, kFloat64, 3},
                                        Params{false, kFloat16, 11}));
}  // namespace mindspore::graphkernel::test
