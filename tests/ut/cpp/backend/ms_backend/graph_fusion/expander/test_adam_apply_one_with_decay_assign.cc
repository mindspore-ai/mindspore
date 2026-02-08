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
  const std::vector<TypePtr> tensor_type;
  TypePtr const_type;
  bool const_is_scalar;
  bool has_assign;
};
}  // namespace

/// Feature: Test AdamApplyOneWithDecayAssign expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestAdamApplyOneWithDecayAssignExpander : public TestGraphKernelExpander,
                                                public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestAdamApplyOneWithDecayAssignExpander, adam_apply_one_with_decay_assign) {
  const auto &param = GetParam();
  ConstructGraph c;
  float beta1_value = 0.9;
  float beta2_value = 0.999;
  auto grad = c.NewTensorInput("grad", param.tensor_type[0], {32, 32});
  auto v = c.NewTensorInput("v", param.tensor_type[1], {32, 32});
  auto m = c.NewTensorInput("m", param.tensor_type[2], {32, 32});
  auto var = c.NewTensorInput("var", param.tensor_type[3], {32, 32});
  auto lr = c.NewValueNode(NewScalar(param.const_type, 0.001, param.const_is_scalar));
  auto beta1 = c.NewValueNode(NewScalar(param.const_type, beta1_value, param.const_is_scalar));
  auto beta1_apply_one = c.NewValueNode(NewScalar(param.const_type, 1.0f - beta1_value, param.const_is_scalar));
  auto beta2 = c.NewValueNode(NewScalar(param.const_type, beta2_value, param.const_is_scalar));
  auto beta2_apply_one = c.NewValueNode(NewScalar(param.const_type, 1.0f - beta2_value, param.const_is_scalar));
  auto decay = c.NewValueNode(NewScalar(param.const_type, 0.0, param.const_is_scalar));
  auto epsilon = c.NewValueNode(NewScalar(param.const_type, 1e-8, param.const_is_scalar));
  CNodePtr op;
  if (param.has_assign) {
    op = c.NewCNodeWithBuildInfo("AdamApplyOneWithDecayAssign",
                                 {grad, v, m, var, lr, beta1, beta1_apply_one, beta2, beta2_apply_one, decay, epsilon});
  } else {
    op = c.NewCNodeWithBuildInfo("AdamApplyOneWithDecay",
                                 {grad, v, m, var, lr, beta1, beta1_apply_one, beta2, beta2_apply_one, decay, epsilon});
  }
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(
  TestOpAdamApplyOneWithDecayAssign, TestAdamApplyOneWithDecayAssignExpander,
  testing::Values(Params{true, {kFloat16, kFloat16, kFloat16, kFloat16}, kFloat16, false, false},
                  Params{true, {kFloat32, kFloat32, kFloat32, kFloat32}, kFloat32, false, false},
                  Params{true, {kBFloat16, kBFloat16, kBFloat16, kBFloat16}, kBFloat16, false, false},
                  Params{true, {kFloat16, kFloat16, kFloat16, kFloat16}, kFloat16, false, true},
                  Params{true, {kFloat32, kFloat32, kFloat32, kFloat32}, kFloat32, false, true},
                  Params{true, {kBFloat16, kBFloat16, kBFloat16, kBFloat16}, kBFloat16, false, true}));
}  // namespace mindspore::graphkernel::test
