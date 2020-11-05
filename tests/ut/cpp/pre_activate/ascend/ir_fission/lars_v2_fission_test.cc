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
#include "backend/optimizer/ascend/ir_fission/lars_v2_fission.h"

namespace mindspore {
namespace opt {
class TestHWLarsV2Fission : public BackendCommon {
 public:
  TestHWLarsV2Fission() : get_py_fun_("gtest_input.pre_activate.lars_v2_fission_test", true) {}
  ~TestHWLarsV2Fission() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWLarsV2Fission, test_fission) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_lars_v2_fission", "before");
  EXPECT_NE(g, nullptr);

  // set abstract for all nodes in g
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  g->get_return()->input(1)->set_abstract(x_abstract);
  for (auto &p: g->parameters()){
    p->set_abstract(x_abstract);
  }
  AbstractBasePtrList args_spec_list;
  auto kg = GetKernelGraph(g, args_spec_list, false);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::LarsV2Fission>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_lars_v2_fission", "after");
  EXPECT_NE(g_after, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
