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
  std::vector<TypePtr> tensor_type;
  TypePtr const_type;
  bool const_is_scalar;
};
}  // namespace

/// Feature: Test LambApplyOptimizerAssign expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestLambApplyOptimizerAssignExpander : public TestGraphKernelExpander,
                                             public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestLambApplyOptimizerAssignExpander, lamb_apply_optimizer_assign) {
  const auto &param = GetParam();
  ConstructGraph c;
  float beta1_value = 0.9;
  float beta2_value = 0.999;
  ShapeVector shape{32, 32};
  auto v_type = param.tensor_type[1];
  auto m_type = param.tensor_type[2];
  auto grad = c.NewTensorInput("grad", param.tensor_type[0], shape);
  auto v = c.NewTensorInput("v", v_type, shape);
  auto m = c.NewTensorInput("m", m_type, shape);
  auto var = c.NewTensorInput("var", param.tensor_type[3], shape);
  auto beta1 = c.NewValueNode(NewScalar(param.const_type, beta1_value, param.const_is_scalar));
  auto beta1_apply_one = c.NewValueNode(NewScalar(param.const_type, 1.0f - beta1_value, param.const_is_scalar));
  auto beta2 = c.NewValueNode(NewScalar(param.const_type, beta2_value, param.const_is_scalar));
  auto beta2_apply_one = c.NewValueNode(NewScalar(param.const_type, 1.0f - beta2_value, param.const_is_scalar));
  auto epsilon = c.NewValueNode(NewScalar(param.const_type, 1e-8, param.const_is_scalar));
  auto step = c.NewValueNode(NewScalar(param.const_type, 10.0, param.const_is_scalar));
  auto do_use_weight = c.NewValueNode(NewScalar(param.const_type, 0.0, param.const_is_scalar));
  auto weight_decay_rate = c.NewValueNode(NewScalar(param.const_type, 0.0, param.const_is_scalar));
  auto u = c.NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto op =
    c.NewCNodeWithoutInfer("LambApplyOptimizerAssign", {grad, v, m, var, beta1, beta1_apply_one, beta2, beta2_apply_one,
                                                        epsilon, step, do_use_weight, weight_decay_rate, u});
  AbstractBasePtrList abs_list{std::make_shared<abstract::AbstractTensor>(m_type, shape),
                               std::make_shared<abstract::AbstractTensor>(v_type, shape),
                               std::make_shared<abstract::AbstractTensor>(m_type, shape)};
  op->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  c.SetGeneralBuildInfo(op);
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpLambApplyOptimizerAssign, TestLambApplyOptimizerAssignExpander,
                        testing::Values(Params{true, {kFloat32, kFloat32, kFloat32, kFloat32}, kFloat32, true},
                                        Params{true, {kFloat16, kFloat16, kFloat16, kFloat16}, kFloat16, false},
                                        Params{true, {kFloat32, kFloat32, kFloat32, kFloat32}, kFloat32, false},
                                        Params{true, {kBFloat16, kBFloat16, kBFloat16, kBFloat16}, kBFloat16, false},
                                        Params{false, {kFloat16, kFloat16, kFloat16, kFloat16}, kFloat32, true},
                                        Params{false, {kBFloat16, kBFloat16, kBFloat16, kBFloat16}, kFloat32, true},
                                        Params{false, {kFloat16, kFloat32, kFloat16, kFloat16}, kFloat32, true}));
}  // namespace mindspore::graphkernel::test
