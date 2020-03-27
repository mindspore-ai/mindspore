/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "session/anf_runtime_algorithm.h"
#include "debug/anf_ir_dump.h"
#include "kernel/kernel_build_info.h"

#define private public
#define protected public
#include "pre_activate/ascend/ir_fusion/conv_bn_relu_fusion.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWConvBnReluFusion : public BackendCommon {
 public:
  TestHWConvBnReluFusion() : get_py_fun_("gtest_input.pre_activate.conv_bn_relu_fusion", true) {}
  ~TestHWConvBnReluFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWConvBnReluFusion, test_conv_bn_relu_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_conv_bn_relu_fusion", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{32, 3, 224, 224};
  std::vector<int> shp_w{64, 3, 7, 7};
  std::vector<int> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto w_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_w);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, w_abstract, b_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);

  // do bn_grad_split_pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::ConvBnReluFusion>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_conv_bn_relu_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
