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
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ir_fusion/adam_apply_one_with_decay_rule.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestHWOptimizeAdamApplyOneWithDecayRuleDyn : public BackendCommon {
 public:
  TestHWOptimizeAdamApplyOneWithDecayRuleDyn()
      : get_py_fun_("gtest_input.pre_activate.adam_apply_one_with_decay_rule_dyn_test", true) {}
  ~TestHWOptimizeAdamApplyOneWithDecayRuleDyn() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

AbstractBasePtr GetAdamApplyOneWithDecayInputAbstract() {
  std::vector<int64_t> shp{-1, 32, -1, 224};
  std::vector<int64_t> max_shp{2, 32, 224, 224};
  auto input_shp = std::make_shared<abstract::Shape>(shp, max_shp);
  auto element = std::make_shared<abstract::AbstractScalar>(kValueAny, std::make_shared<Float>(32));
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(element, input_shp);
  return x_abstract;
}

/// Feature: test AdamApplyOneWithDecay dynamic shape
/// Description:  The input shape is dynamic
/// Expectation: Assert that result is error
TEST_F(TestHWOptimizeAdamApplyOneWithDecayRuleDyn, test_adam_apply_one_with_decay_rule_dyn_cond1) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "before_cond1");
  auto x_abstract = GetAdamApplyOneWithDecayInputAbstract();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 11; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneWithDecayRuleCond1>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

/// Feature: test AdamApplyOneWithDecay dynamic shape
/// Description:  The input shape is dynamic
/// Expectation: Assert that result is error
TEST_F(TestHWOptimizeAdamApplyOneWithDecayRuleDyn, test_adam_apply_one_with_decay_rule_dyn_cond2) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "before_cond2");
  auto x_abstract = GetAdamApplyOneWithDecayInputAbstract();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 11; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneWithDecayRuleCond2>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

/// Feature: test AdamApplyOneWithDecay dynamic shape
/// Description:  The input shape is dynamic
/// Expectation: Assert that result is error
TEST_F(TestHWOptimizeAdamApplyOneWithDecayRuleDyn, test_adam_apply_one_with_decay_rule_dyn_cond3) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "before_cond3");
  auto x_abstract = GetAdamApplyOneWithDecayInputAbstract();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 11; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneWithDecayRuleCond3>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

/// Feature: test AdamApplyOneWithDecay dynamic shape
/// Description:  The input shape is dynamic
/// Expectation: Assert that result is error
TEST_F(TestHWOptimizeAdamApplyOneWithDecayRuleDyn, test_adam_apply_one_with_decay_rule_dyn_cond4) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "before_cond4");
  auto x_abstract = GetAdamApplyOneWithDecayInputAbstract();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 11; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneWithDecayRuleCond4>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

/// Feature: test AdamApplyOneWithDecay dynamic shape
/// Description:  The input shape is dynamic
/// Expectation: Assert that result is error
TEST_F(TestHWOptimizeAdamApplyOneWithDecayRuleDyn, test_adam_apply_one_with_decay_rule_dyn_cond5) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "before_cond5");
  auto x_abstract = GetAdamApplyOneWithDecayInputAbstract();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 11; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AdamApplyOneWithDecayRuleCond5>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_adam_apply_one_with_decay_rule_dyn", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore