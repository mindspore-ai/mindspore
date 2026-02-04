/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <limits>
#include <string>

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "utils/ms_context.h"
#include "common/graph_optimizer_test_framework.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ms_backend/graph_fusion/core/graph_kernel_pass_manager.h"
#include "include/backend/common/pass_manager/optimizer.h"
#include "ir/anf.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "include/utils/anfalgo.h"
#include "utils/phase.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_layer_norm_fusion.h"

namespace mindspore::graphkernel::test {
namespace {
opt::PassManagerPtr GetControllablePassManager() {
  auto pm = std::make_shared<graphkernel::GraphKernelPassManager>(std::numeric_limits<size_t>::max(), "test_controll");
  pm->Add(std::make_shared<opt::AddLayernormFusion>(), graphkernel::OptLevel_0);
  return pm;
}
}  // namespace

struct PassSwitchParams {
  bool switch_off;
};

/// Feature: Test graph kernel fusion pass switch.
/// Description: Register fusion pass to graph kernel, switch on or off by flags.
/// Expectation: Get expected passes by type, control the passes by flags.
class TestPassSwitch : public GraphKernelCommonTestSuite, public testing::WithParamInterface<PassSwitchParams> {
 public:
  void SetUp() override {
    const auto &param = GetParam();
    bool switch_off = param.switch_off;
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
    context->SetMsInternalEnableCustomKernelList();

    if (switch_off) {
      std::map<std::string, std::string> gk_jit_config;
      gk_jit_config["graph_kernel_flags"] = "--disable_pass=add_layer_norm_fusion";
      graphkernel::GraphKernelFlags::SaveJitConfig(gk_jit_config);
    } else {
      std::map<std::string, std::string> gk_jit_config;
      gk_jit_config["graph_kernel_flags"] = "";
      graphkernel::GraphKernelFlags::SaveJitConfig(gk_jit_config);
    }
  }
};

TEST_P(TestPassSwitch, pass_switch) {
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

  auto graph = c.GetGraph();
  UT_CHECK_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = GetControllablePassManager();
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);

  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddVar("gamma").AddVar("beta").AddVar("eps").AddCNode(
    "add_layer_norm", {std::make_shared<Primitive>("AddLayerNorm"), "input_0", "input_1", "gamma", "beta", "eps"});

  const auto &param = GetParam();
  if (param.switch_off) {
    EXPECT_FALSE(checker.build_pattern_map(c.GetGraph()->output()));
  } else {
    EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
  }
}

INSTANTIATE_TEST_CASE_P(TestPassSwitchCases, TestPassSwitch,
                        testing::Values(PassSwitchParams{true}, PassSwitchParams{false}));
}  // namespace mindspore::graphkernel::test
