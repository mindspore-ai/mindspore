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
#include "pre_activate/ascend/ir_fusion/lamb_next_right_rule.h"

namespace mindspore {
namespace opt {

class TestHWLambNextRightRule : public BackendCommon {
 public:
  TestHWLambNextRightRule() : get_py_fun_("gtest_input.pre_activate.lamb_next_right_rule_test", true) {}
  ~TestHWLambNextRightRule() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWLambNextRightRule, test_lamb_next_right_rule_matched) {
  /*
   * def before(input0, input1, mul2_x, mul3_x, true_div1_recip, add2_y):
   * square0 = Square(input0)
   * mul2 = Mul(mul2_x, input1)
   * mul3 = Mul(mul3_x, square0)
   * add1 = Add(mul2, mul3)
   * real_div1 = Mul(add1, true_div1_recip)
   * add2 = Add(sqrt0, add2_y)
   * outputs = make_tuple(add1, add2)
   * output = tuple_getitem(outputs, 0)
   * return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_right_rule", "before");
  std::vector<int> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextRightRule>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_next_right_rule", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLambNextRightRule, test_lamb_next_right_rule_unmatched) {
  /*
   * def before(input0, input1, mul2_x, mul3_x, true_div1_recip, add2_y):
   * square0 = Square(input0)
   * mul2 = Mul(mul2_x, input1)
   * mul3 = Add(mul3_x, square0)
   * add1 = Add(mul2, mul3)
   * real_div1 = Mul(add1, true_div1_recip)
   * add2 = Add(sqrt0, add2_y)
   * outputs = make_tuple(add1, add2)
   * output = tuple_getitem(outputs, 0)
   * return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_next_right_rule", "before_unmatched");
  std::vector<int> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 6; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambNextRightRule>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
}
}  // namespace opt
}  // namespace mindspore