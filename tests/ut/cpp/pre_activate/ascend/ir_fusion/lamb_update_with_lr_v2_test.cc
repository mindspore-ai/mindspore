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
#include "debug/anf_ir_dump.h"
#include "backend/optimizer/ascend/ir_fusion/lamb_update_with_lr_v2.h"

namespace mindspore {
namespace opt {

class TestHWLambUpdateWithLrV2 : public BackendCommon {
 public:
  TestHWLambUpdateWithLrV2() : get_py_fun_("gtest_input.pre_activate.lamb_update_with_lr_v2_test", true) {}
  ~TestHWLambUpdateWithLrV2() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWLambUpdateWithLrV2, test_lamb_update_with_lr_v2) {
  /*
   * def before(input0, input1, input2, input3, input4, select_e, greater_y):
   * greater0 = Greater(input0, greater_y)
   * greater1 = Greater(input1, greater_y)
   * real_div0 = RealDiv(input0, input1)
   * select0 = Select(greater1, real_div0, select_e)
   * select1 = Select(greater0, select0, select_e)
   * mul0 = Mul(select1, input2)
   * mul1 = Mul(mul0, input3)
   * sub0 = Sub(input4, mul1)
   * return sub0
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lamb_update_with_lr_v2", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 7; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LambUpdateWithLrV2>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lamb_update_with_lr_v2", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
