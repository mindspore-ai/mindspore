/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/optimize_dependence.h"

namespace mindspore {
namespace opt {

class TestHWOptimizeDependence : public BackendCommon {
 public:
  TestHWOptimizeDependence() : get_py_fun_("gtest_input.pre_activate.optimize_dependence_test", true) {}
  ~TestHWOptimizeDependence() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWOptimizeDependence, test_optimize_dependence) {
  /*
   * def test_eliminate_depend_input2(x, y, z):
   *     new_z = four2five(z)
   *     depend_intput = depend(y, new_z)
   *     sum = add(x, depend_intput)
   *     return sum
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_optimize_dependence", "before");

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::OptimizeDependence>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_optimize_dependence", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWOptimizeDependence, test_optimize_dependence_with_make_tuple) {
  /*
   * def before(x, y, a, b):
   *    z = make_tuple(TransData(a), TransData(b))
   *    depend_intput = depend(y, z)
   *    sum = add(x, depend_intput)
   *    return sum
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_optimize_dependence_with_make_tuple", "before");

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::OptimizeDependence>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_optimize_dependence_with_make_tuple", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}


TEST_F(TestHWOptimizeDependence, test_optimize_control_dependence_with_make_tuple) {
  /*
   * def before(x, y, a, b):
   *    z = make_tuple(TransData(a), TransData(b))
   *    depend_intput = depend(y, z)
   *    sum_add = add(x, depend_intput)
   *    return sum_add
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_optimize_control_dependence_with_make_tuple", "before");

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::OptimizeDependence>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_optimize_control_dependence_with_make_tuple", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}


TEST_F(TestHWOptimizeDependence, test_optimize_control_dependence) {
  /*
   * def before(x, y, z):
   *    new_z = TransData(z)
   *    depend_intput = depend(y, new_z)
   *    sum_add = add(x, depend_intput)
   *    return sum_add
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_optimize_control_dependence", "before");

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::OptimizeDependence>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_optimize_control_dependence", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
