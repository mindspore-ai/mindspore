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

/// Feature: Test LambApplyWeightAssign expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestLambApplyWeightAssignExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestLambApplyWeightAssignExpander, lamb_apply_weight_assign) {
  const auto &param = GetParam();
  ConstructGraph c;
  ShapeVector shape{32, 32};
  auto var_type = param.tensor_type[3];
  auto w_norm = c.NewTensorInput("w_norm", param.tensor_type[0], {1});
  auto g_norm = c.NewTensorInput("g_norm", param.tensor_type[1], {1});
  auto update = c.NewTensorInput("update", param.tensor_type[2], shape);
  auto var = c.NewTensorInput("var", var_type, shape);
  auto lr = c.NewValueNode(NewScalar(param.const_type, 0.01, param.const_is_scalar));
  auto u = c.NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto op = c.NewCNodeWithoutInfer("LambApplyWeightAssign", {w_norm, g_norm, lr, update, var, u});
  op->set_abstract(std::make_shared<abstract::AbstractTensor>(var_type, shape));
  c.SetGeneralBuildInfo(op);
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(TestOpLambApplyWeightAssign, TestLambApplyWeightAssignExpander,
                        testing::Values(Params{true, {kFloat32, kFloat32, kFloat32, kFloat32}, kFloat32, true},
                                        Params{true, {kFloat16, kFloat16, kFloat16, kFloat16}, kFloat16, false},
                                        Params{true, {kFloat32, kFloat32, kFloat32, kFloat32}, kFloat32, false},
                                        Params{false, {kFloat16, kFloat32, kFloat32, kFloat32}, kFloat32, true},
                                        Params{false, {kFloat16, kFloat16, kFloat16, kFloat16}, kFloat32, true},
                                        Params{false, {kBFloat16, kBFloat16, kBFloat16, kBFloat16}, kFloat32, true},
                                        Params{false, {kBFloat16, kBFloat16, kBFloat16, kBFloat16}, kBFloat16, false}));
}  // namespace mindspore::graphkernel::test
