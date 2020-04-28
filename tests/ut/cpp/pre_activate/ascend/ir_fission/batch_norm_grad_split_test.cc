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
#include "operator/ops.h"
#include "ir/meta_tensor.h"
#include "debug/anf_ir_dump.h"
#include "utils/utils.h"
#include "pre_activate/common/optimizer.h"
#include "pre_activate/ascend/ir_fission/batch_norm_grad_split.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
class TestHWBatchNormGradSplit : public BackendCommon {
 public:
  TestHWBatchNormGradSplit() : get_py_fun_("gtest_input.pre_activate.batch_norm_grad_split", true) {}

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWBatchNormGradSplit, test_split) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_batch_norm_grad_split", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  std::vector<int> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, b_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kernel_graph, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::BatchNormGradSplit>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_batch_norm_grad_split", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
