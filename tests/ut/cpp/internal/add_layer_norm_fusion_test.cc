/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/graph_optimizer_test_framework.h"
#include "primitive/sequence_ops.h"
#include "common/common_test.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_layer_norm_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "utils/phase.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

namespace mindspore {
namespace opt {
class AddLayernormFusionUT : public UT::Common {
 public:
  AddLayernormFusionUT() {}
};

/// Feature: A backend pass: AddLayernormFusion
/// Description: Convert LayerNorm(Add) to AddLayernorm
/// Expectation: After optimize, match AddLayernorm.
TEST_F(AddLayernormFusionUT, AddLayernormFusionTest) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kFloat16, {1, 1024, 11264});
  auto input_1 = c.NewTensorInput("input_1", kFloat16, {1, 1024, 11264});
  auto gamma = c.NewTensorInput("gamma", kFloat16, {11264});
  auto beta = c.NewTensorInput("beta", kFloat16, {11264});
  auto add = c.NewCNode("Add", {input_0, input_1}, {});
  auto begin_norm_axis = c.NewValueNode(MakeValue<int64_t>(-1));
  auto begin_params_axis = c.NewValueNode(MakeValue<int64_t>(-1));
  auto eps = c.NewValueNode(MakeValue<float>(1e-5));
  auto layernorm = c.NewCNode("LayerNorm", {add, gamma, beta, begin_norm_axis, begin_params_axis, eps}, {});
  c.SetOutput(layernorm);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AddLayernormFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddVar("gamma").AddVar("beta").AddVar("eps").AddCNode(
    "add_layer_norm", {std::make_shared<Primitive>("AddLayerNorm"), "input_0", "input_1", "gamma", "beta", "eps"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
/// Feature: A backend pass: AddLayernormFusion
/// Description: Convert LayerNormV3(Add) to AddLayernorm
/// Expectation: After optimize, match AddLayernorm.
TEST_F(AddLayernormFusionUT, AddLayernormV3FusionTest) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kFloat16, {1, 1024, 11264});
  auto input_1 = c.NewTensorInput("input_1", kFloat16, {1, 1024, 11264});
  auto gamma = c.NewTensorInput("gamma", kFloat16, {11264});
  auto beta = c.NewTensorInput("beta", kFloat16, {11264});
  auto add = c.NewCNode("Add", {input_0, input_1}, {});
  auto begin_norm_axis = c.NewValueNode(MakeValue<int64_t>(-1));
  auto begin_params_axis = c.NewValueNode(MakeValue<int64_t>(-1));
  auto eps = c.NewValueNode(MakeValue<float>(1e-5));
  auto layernorm = c.NewCNode("LayerNormV3", {add, gamma, beta, begin_norm_axis, begin_params_axis, eps}, {});
  c.SetOutput(layernorm);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AddLayernormV3Fusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddVar("gamma").AddVar("beta").AddVar("eps").AddCNode(
    "add_layer_norm", {std::make_shared<Primitive>("AddLayerNorm"), "input_0", "input_1", "gamma", "beta", "eps"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: AddLayernormFusion
/// Description: Convert LayerNormExt(Add) to AddLayernorm
/// Expectation: After optimize, match AddLayernorm.
TEST_F(AddLayernormFusionUT, AddLayernormExtFusionTest) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kFloat16, {1, 1024, 11264});
  auto input_1 = c.NewTensorInput("input_1", kFloat16, {1, 1024, 11264});
  auto gamma = c.NewTensorInput("gamma", kFloat16, {11264});
  auto beta = c.NewTensorInput("beta", kFloat16, {11264});
  auto add = c.NewCNode("Add", {input_0, input_1}, {});
  auto normalize_shape = c.NewValueNode(MakeValue<std::vector<int64_t>>({11264}));
  auto eps = c.NewValueNode(MakeValue<float>(1e-5));
  auto layernorm = c.NewCNode("LayerNormExt", {add, normalize_shape, gamma, beta, eps}, {});
  c.SetOutput(layernorm);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AddLayernormExtFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddVar("gamma").AddVar("beta").AddVar("eps").AddCNode(
    "add_layer_norm", {std::make_shared<Primitive>("AddLayerNorm"), "input_0", "input_1", "gamma", "beta", "eps"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace opt
}  // namespace mindspore
