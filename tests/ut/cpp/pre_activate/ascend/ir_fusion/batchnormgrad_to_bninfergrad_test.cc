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
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ir_fusion/batchnormgrad_to_bninfergrad.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestHWOptimizeBatchNormGrad2BNInferGrad : public BackendCommon {
 public:
  TestHWOptimizeBatchNormGrad2BNInferGrad()
      : get_py_fun_("gtest_input.pre_activate.batchnormgrad_to_bninfergrad", true) {}
  ~TestHWOptimizeBatchNormGrad2BNInferGrad() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWOptimizeBatchNormGrad2BNInferGrad, test_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batchnormgrad_to_bninfergrad", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, y_abstract, y_abstract, y_abstract, y_abstract};
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormGrad2BNInferGrad>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_batchnormgrad_to_bninfergrad", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWOptimizeBatchNormGrad2BNInferGrad, test_no_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batchnormgrad_to_bninfergrad", "no_fusion");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, y_abstract, y_abstract, y_abstract, y_abstract};
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNormGrad2BNInferGrad>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
}
}  // namespace opt
}  // namespace mindspore
