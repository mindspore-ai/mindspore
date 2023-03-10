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
#include "plugin/device/ascend/optimizer/ir_fusion/batchnorm_to_bninfer.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestHWOptimizeBatchNorm2BNInfer : public BackendCommon {
 public:
  TestHWOptimizeBatchNorm2BNInfer() : get_py_fun_("gtest_input.pre_activate.batchnorm_to_bninfer", true) {}
  ~TestHWOptimizeBatchNorm2BNInfer() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWOptimizeBatchNorm2BNInfer, test_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batchnorm_to_bninfer", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract, y_abstract, y_abstract, y_abstract};
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNorm2BNInfer>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_batchnorm_to_bninfer", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWOptimizeBatchNorm2BNInfer, test_no_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batchnorm_to_bninfer", "no_fusion");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{32, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  std::vector<int64_t> shp_y{64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_y);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract, y_abstract, y_abstract, y_abstract};
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_graph = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::BatchNorm2BNInfer>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
}
}  // namespace opt
}  // namespace mindspore
