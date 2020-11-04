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
#define private public
#define protected public
#include "backend/optimizer/ascend/ir_fission/split_fission.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWSplitFission : public BackendCommon {
 public:
  TestHWSplitFission() : get_py_fun_("gtest_input.pre_activate.split_fission_test", true) {}
  ~TestHWSplitFission() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWSplitFission, test_split_fission_divided_by_3) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_split_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{512, 3, 1};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  args_spec_list.push_back(x_abstract);
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto split_fission = std::make_shared<opt::SplitFission>();
  split_fission->outputs_divisor_ = 3;
  pm->AddPass(split_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_split_fission", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
