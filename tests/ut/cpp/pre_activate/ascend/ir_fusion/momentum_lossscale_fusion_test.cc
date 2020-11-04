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

#include "backend/optimizer/ascend/ir_fusion/momentum_lossscale_fusion.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"

namespace mindspore {
namespace opt {
class TestHWMomentumLossscaleFusion : public BackendCommon {
 public:
  TestHWMomentumLossscaleFusion() : get_py_fun_("gtest_input.pre_activate.momentum_lossscale_fusion_test", true) {}
  ~TestHWMomentumLossscaleFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWMomentumLossscaleFusion, test_momentum_lossscale_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_momentum_lossscale_fusion", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 5; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::MomentumLossscaleFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_momentum_lossscale_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore