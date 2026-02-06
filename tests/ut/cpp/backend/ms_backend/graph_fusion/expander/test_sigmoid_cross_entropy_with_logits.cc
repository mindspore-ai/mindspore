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
  std::vector<TypePtr> inputs_type;
  ShapeArray inputs_shape;
  bool is_grad;
};
}  // namespace

/// Feature: Test graph kernel SigmoidCrossEntropyWithLogits/SigmoidCrossEntropyWithLogitsGrad expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestSigmoidCrossEntropyWithLogitsExpander : public TestGraphKernelExpander,
                                                  public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestSigmoidCrossEntropyWithLogitsExpander, sigmoid_cross_entropy_with_logits) {
  const auto &param = GetParam();
  ConstructGraph c;
  std::vector<AnfNodePtr> inputs(param.inputs_type.size());
  for (size_t i = 0; i < param.inputs_type.size(); ++i) {
    inputs[i] = c.NewTensorInput("x" + std::to_string(i), param.inputs_type[i], param.inputs_shape[i]);
  }
  auto op = param.is_grad ? c.NewCNodeWithBuildInfo("SigmoidCrossEntropyWithLogitsGrad", inputs)
                          : c.NewCNodeWithBuildInfo("SigmoidCrossEntropyWithLogits", inputs);
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(c.GetGraph()).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpSigmoid, TestSigmoidCrossEntropyWithLogitsExpander,
                        testing::Values(
                          // SigmoidCrossEntropyWithLogits
                          Params{true, {kFloat16, kFloat16}, {{2, 3}, {2, 3}}, false},
                          Params{true, {kFloat32, kFloat32}, {{2, 3}, {2, 3}}, false},
                          Params{false, {kFloat64, kFloat64}, {{2, 3}, {2, 3}}, false},
                          Params{false, {kInt8, kInt8}, {{2, 3}, {2, 3}}, false},
                          Params{false, {kInt32, kInt32}, {{2, 3}, {2, 3}}, false},
                          Params{false, {kInt64, kInt64}, {{2, 3}, {2, 3}, {2, 3}}, false},
                          // SigmoidCrossEntropyWithLogitsGrad
                          Params{true, {kFloat16, kFloat16, kFloat16}, {{2, 3}, {2, 3}, {2, 3}}, true},
                          Params{true, {kFloat32, kFloat32, kFloat32}, {{2, 3}, {2, 3}, {2, 3}}, true},
                          Params{false, {kFloat64, kFloat64, kFloat64}, {{2, 3}, {2, 3}, {2, 3}}, true},
                          Params{false, {kInt8, kInt8, kInt8}, {{2, 3}, {2, 3}, {2, 3}}, true},
                          Params{false, {kInt32, kInt32, kInt32}, {{2, 3}, {2, 3}, {2, 3}}, true},
                          Params{false, {kInt64, kInt64, kInt64}, {{2, 3}, {2, 3}, {2, 3}}, true}));
}  // namespace mindspore::graphkernel::test