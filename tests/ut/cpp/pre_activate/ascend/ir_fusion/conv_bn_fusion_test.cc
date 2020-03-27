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
#include "operator/ops.h"
#include "debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "pre_activate/common/optimizer.h"
#include "pre_activate/common/pass_manager.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_info.h"

#define private public
#define protected public
#include "pre_activate/ascend/ir_fusion/conv_bn_fusion.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestHWConvBnFusion : public BackendCommon {
 public:
  TestHWConvBnFusion() : getPyFun_("gtest_input.pre_activate.ir_fusion_test", true) {}
  ~TestHWConvBnFusion() override = default;

  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestHWConvBnFusion, test_conv_bn_fusion) {
  /*
   * def before(x, y):
   *    conv_output = conv(x, y)
   *    bn_output = bn(conv_output)
   *    item0 = tuple_getitem(bn_output, 0)
   *    item1 = tuple_getitem(bn_output, 3)
   *    item2 = tuple_getitem(bn_output, 4)
   *    res = make_tuple(item0, item1, item2)
   *    return res
   */
  getPyFun_.SetDoResolve(true);
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_conv_bn_fusion", "before");
  std::vector<int> shp_x{32, 3, 224, 224};
  std::vector<int> shp_w{64, 3, 7, 7};
  std::vector<int> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto w_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_w);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, w_abstract, b_abstract, b_abstract, b_abstract, b_abstract};
  auto fg = GetKernelGraph(g, args_spec_list);

  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  auto conv_bn_fusion_pass = std::make_shared<opt::ConvBnFusion>();
  pass_manager->AddPass(conv_bn_fusion_pass);
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g = graph_optimizer->Optimize(fg);

  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_conv_bn_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_g));
}

}  // namespace opt
}  // namespace mindspore