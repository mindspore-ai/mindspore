/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/ir_fusion/lamb_next_mv_with_decay_rule.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {

class TestHWLambNextMVWithDecayRule : public BackendCommon {
 public:
  TestHWLambNextMVWithDecayRule() : get_py_fun_("gtest_input.pre_activate.lamb_next_mv_with_decay_rule_test", true) {}
  ~TestHWLambNextMVWithDecayRule() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_decay_rule_cond4_matched) {
  /*
   * def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
   * constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
   * mul1 = Mul(constant_mul1_sub, input3)
   * mul0 = Mul(constant_mul0_x, input4)
   * add0 = Add(mul0, mul1)
   * mul2 = Mul(constant_mul2_x, input1)
   * mul3 = Mul(constant_mul3_sub1, input0)
   * add1 = Add(mul2, mul3)
   * real_div1 = RealDiv(add1, input2)
   * add2 = Add(real_div1, constant_add2_y)
   * sqrt1 = Sqrt(real_div1)
   * real_div0 = RealDiv(add0, input5)
   * add4 = Add(sqrt1, constant_add2_y)
   * sqrt0 = Rsqrt(add2)
   * mul4 = Mul(constant_mul4_x, input6)
   * real_div4 = RealDiv(real_div0, add4)
   * real_div2 = Mul(real_div0, sqrt0)
   * add5 = Add(real_div4, mul4)
   * add3 = Add(real_div2, mul4)
   * outputs = make_tuple(add3, add0, add1, add5)
   * output = tuple_getitem(outputs, 0)
   * return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond4>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_decay_rule_cond4_unmatched_add3) {
  /*
   * def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
   * constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
   * mul1 = Mul(constant_mul1_sub, input3)
   * mul0 = Mul(constant_mul0_x, input4)
   * add0 = Add(mul0, mul1)
   * mul2 = Mul(constant_mul2_x, input1)
   * mul3 = Mul(constant_mul3_sub1, input0)
   * add1 = Add(mul2, mul3)
   * real_div1 = RealDiv(add1, input2)
   * add2 = Add(real_div1, constant_add2_y)
   * sqrt1 = Sqrt(real_div1)
   * real_div0 = RealDiv(add0, input5)
   * add4 = Add(sqrt1, constant_add2_y)
   * sqrt0 = Rsqrt(add2)
   * mul4 = Mul(constant_mul4_x, input6)
   * real_div4 = RealDiv(real_div0, add4)
   * real_div2 = Mul(real_div0, sqrt0)
   * add5 = Add(real_div4, mul4)
   * add3 = Mul(real_div2, mul4)
   * outputs = make_tuple(add3, add0, add1, add5)
   * output = tuple_getitem(outputs, 0)
   * return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "before_unmatched_add3");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond4>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_decay_rule_cond4_unmatched_mul4) {
  /*
   * def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
   * constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
   * mul1 = Mul(constant_mul1_sub, input3)
   * mul0 = Mul(constant_mul0_x, input4)
   * add0 = Add(mul0, mul1)
   * mul2 = Mul(constant_mul2_x, input1)
   * mul3 = Mul(constant_mul3_sub1, input0)
   * add1 = Add(mul2, mul3)
   * real_div1 = RealDiv(add1, input2)
   * add2 = Add(real_div1, constant_add2_y)
   * sqrt1 = Sqrt(real_div1)
   * real_div0 = RealDiv(add0, input5)
   * add4 = Add(sqrt1, constant_add2_y)
   * sqrt0 = Rsqrt(add2)
   * mul4 = Add(constant_mul4_x, input6)
   * real_div4 = RealDiv(real_div0, add4)
   * real_div2 = Mul(real_div0, sqrt0)
   * add5 = Add(real_div4, mul4)
   * add3 = Add(real_div2, mul4)
   * outputs = make_tuple(add3, add0, add1, add5)
   * output = tuple_getitem(outputs, 0)
   * return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "before_unmatched_mul4");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond4>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_decay_rule_cond4_unmatched_real_div0) {
  /*
   * def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
   * constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
   * mul1 = Mul(constant_mul1_sub, input3)
   * mul0 = Mul(constant_mul0_x, input4)
   * add0 = Add(mul0, mul1)
   * mul2 = Mul(constant_mul2_x, input1)
   * mul3 = Mul(constant_mul3_sub1, input0)
   * add1 = Add(mul2, mul3)
   * real_div1 = RealDiv(add1, input2)
   * add2 = Add(real_div1, constant_add2_y)
   * sqrt1 = Sqrt(real_div1)
   * real_div0 = Add(add0, input5)
   * add4 = Add(sqrt1, constant_add2_y)
   * sqrt0 = Rsqrt(add2)
   * mul4 = Mul(constant_mul4_x, input6)
   * real_div4 = RealDiv(real_div0, add4)
   * real_div2 = Mul(real_div0, sqrt0)
   * add5 = Add(real_div4, mul4)
   * add3 = Add(real_div2, mul4)
   * outputs = make_tuple(add3, add0, add1, add5)
   * output = tuple_getitem(outputs, 0)
   * return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "before_unmatched_real_div0");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond4>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_decay_rule_cond4_unmatched_real_div1) {
  /*
   * def before(input0, input1, input2, input3, input4, input5, input6, constant_mul0_x, constant_mul1_sub,
   * constant_mul2_x, constant_mul3_sub1, constant_mul4_x, constant_add2_y):
   * mul1 = Mul(constant_mul1_sub, input3)
   * mul0 = Mul(constant_mul0_x, input4)
   * add0 = Add(mul0, mul1)
   * mul2 = Mul(constant_mul2_x, input1)
   * mul3 = Mul(constant_mul3_sub1, input0)
   * add1 = Add(mul2, mul3)
   * real_div1 = Add(add1, input2)
   * add2 = Add(real_div1, constant_add2_y)
   * sqrt1 = Sqrt(real_div1)
   * real_div0 = RealDiv(add0, input5)
   * add4 = Add(sqrt1, constant_add2_y)
   * sqrt0 = Rsqrt(add2)
   * mul4 = Mul(constant_mul4_x, input6)
   * real_div4 = RealDiv(real_div0, add4)
   * real_div2 = Mul(real_div0, sqrt0)
   * add5 = Add(real_div4, mul4)
   * add3 = Add(real_div2, mul4)
   * outputs = make_tuple(add3, add0, add1, add5)
   * output = tuple_getitem(outputs, 0)
   * return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "before_unmatched_real_div1");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond4>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond4", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_with_decay_rule_cond1) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond1", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond1>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond1", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_with_decay_rule_cond1_un_match) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond1", "un_match");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond1>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond1", "un_match");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_with_decay_rule_cond2) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond2", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond2>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond2", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_with_decay_rule_cond2_un_match) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond2", "un_match");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond2>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond2", "un_match");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_with_decay_rule_cond3) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond3", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond3>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond3", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextMVWithDecayRule, test_lamb_next_mv_with_decay_rule_cond3_un_match) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond3", "un_match");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 13; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextMVWithDecayRuleCond3>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_mv_with_decay_rule_cond3", "un_match");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
