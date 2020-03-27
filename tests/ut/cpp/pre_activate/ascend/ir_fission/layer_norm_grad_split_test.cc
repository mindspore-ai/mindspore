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
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"
#include "ir/meta_tensor.h"
#include "debug/anf_ir_dump.h"
#include "utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "pre_activate/common/optimizer.h"

#define private public
#define protected public
#include "pre_activate/ascend/ir_fission/layer_norm_grad_split.h"
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

class MockLayerNormGradSplitKernelSelect : public KernelSelect {
 public:
  MockLayerNormGradSplitKernelSelect() = default;
  ~MockLayerNormGradSplitKernelSelect() override = default;
  void SelectKernel(const CNodePtr &cnode) override {
    auto name = AnfAlgo::GetCNodeName(cnode);

    if (name == kLayerNormXBackpropOpName) {
      kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
      builder.SetInputsFormat(
        {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
      builder.SetInputsDeviceType(
        {kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
      builder.SetOutputsFormat({kOpFormat_NC1HWC0});
      builder.SetOutputsDeviceType({kNumberTypeFloat16});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
      return;
    }
    if (name == kLayerNormBetaGammaBackpropOpName) {
      kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
      builder.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
      builder.SetOutputsFormat({kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
      builder.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
      return;
    }
  }
};  // namespace opt

TEST_F(TestHWLayerNormGradSplit, test_layer_norm_grad_split) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_layer_norm_grad_split", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  std::vector<int> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kernel_graph, nullptr);

  // get LayerNormGrad
  CNodePtr ret = kernel_graph->get_return();
  EXPECT_NE(ret, nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_TRUE(ret->input(1)->isa<CNode>());
  auto make_tuple1 = ret->input(1)->cast<CNodePtr>();
  EXPECT_NE(make_tuple1->input(1), nullptr);
  EXPECT_TRUE(make_tuple1->input(1)->isa<CNode>());
  auto make_tuple2 = make_tuple1->input(1)->cast<CNodePtr>();
  EXPECT_NE(make_tuple2->input(1), nullptr);
  EXPECT_TRUE(make_tuple2->input(1)->isa<CNode>());
  auto tuple_getitem = make_tuple2->input(1)->cast<CNodePtr>();
  EXPECT_NE(tuple_getitem->input(1), nullptr);
  EXPECT_TRUE(tuple_getitem->input(1)->isa<CNode>());
  auto layer_norm_grad = tuple_getitem->input(1)->cast<CNodePtr>();

  // set kernel for LayerNormGrad
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder1.SetOutputsFormat({kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder1.SetInputsDeviceType(
    {kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
  builder1.SetKernelType(TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), layer_norm_grad.get());

  // get param5
  EXPECT_NE(layer_norm_grad->input(5), nullptr);
  auto param = layer_norm_grad->input(5);

  // set kernel for param5
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetOutputsFormat({kOpFormat_NC1HWC0});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), param.get());

  // do layer_norm_grad_split pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::LayerNormGradSplit>();
  auto kernel_select = std::make_shared<MockLayerNormGradSplitKernelSelect>();
  pass->kernel_select_ = kernel_select;
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_layer_norm_grad_split", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
