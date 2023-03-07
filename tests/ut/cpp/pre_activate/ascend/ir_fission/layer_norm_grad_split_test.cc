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
#include "include/backend/anf_runtime_algorithm.h"
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "backend/common/optimizer/optimizer.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/ir_fission/layer_norm_grad_split.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWLayerNormGradSplit : public BackendCommon {
 public:
  TestHWLayerNormGradSplit() : get_py_fun_("gtest_input.pre_activate.layer_norm_grad_split", true) {}

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWLayerNormGradSplit, test_layer_norm_grad_split) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_layer_norm_grad_split", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  std::vector<int64_t> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kernel_graph, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::LayerNormGradSplit>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_layer_norm_grad_split", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
