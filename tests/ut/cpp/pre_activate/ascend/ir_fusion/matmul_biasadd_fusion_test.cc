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

#include "backend/optimizer/ascend/ir_fusion/matmul_biasadd_fusion.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"

namespace mindspore {
namespace opt {
class TestHWMatmulBiasaddFusion : public BackendCommon {
 public:
  TestHWMatmulBiasaddFusion() : get_py_fun_("gtest_input.pre_activate.matmul_biasadd_fusion_test", true) {}
  ~TestHWMatmulBiasaddFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWMatmulBiasaddFusion, test_matmul_biasadd_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_matmul_biasadd_fusion", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shpx{1, 3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shpx);
  std::vector<int64_t> shpy{3, 4};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shpy);
  std::vector<int64_t> shp_bias{4};
  auto bias_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_bias);
  AbstractBasePtrList args_spec_list;
  args_spec_list.push_back(x_abstract);
  args_spec_list.push_back(y_abstract);
  args_spec_list.push_back(bias_abstract);
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::MatmulBiasaddFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_matmul_biasadd_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore