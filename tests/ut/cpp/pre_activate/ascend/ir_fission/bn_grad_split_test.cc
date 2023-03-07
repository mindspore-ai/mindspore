/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/bn_grad_split.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWBnGradSplit : public BackendCommon {
 public:
  TestHWBnGradSplit() : get_py_fun_("gtest_input.pre_activate.bn_grad_split", true) {}

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWBnGradSplit, test_bn_grad_split_tbe) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_bn_grad_split", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  std::vector<int64_t> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, b_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kernel_graph, nullptr);

  // get BNGrad
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
  auto bn_grad = tuple_getitem->input(1)->cast<CNodePtr>();

  // get param1
  EXPECT_NE(bn_grad->input(1), nullptr);
  auto param1 = bn_grad->input(1);

  // set kernel for param1
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetOutputsFormat({kOpFormat_NC1HWC0});
  builder2.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), param1.get());

  // set kernel for BNGrad
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0,
     kOpFormat_NC1HWC0});
  builder1.SetOutputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0,
     kOpFormat_NC1HWC0});
  builder1.SetInputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32,
     kNumberTypeFloat32});
  builder1.SetOutputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32,
     kNumberTypeFloat32});
  builder1.SetKernelType(TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), bn_grad.get());
  // do bn_grad_split pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::BnGradSplit>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_bn_grad_split", "after2");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWBnGradSplit, test_sync_bn_grad_split_tbe) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_sync_bn_grad_split", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  std::vector<int64_t> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kernel_graph, nullptr);

  // get SyncBNGrad
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
  auto bn_grad = tuple_getitem->input(1)->cast<CNodePtr>();

  // get param1
  EXPECT_NE(bn_grad->input(1), nullptr);
  auto param1 = bn_grad->input(1);

  // set kernel for param1
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetOutputsFormat({kOpFormat_NC1HWC0});
  builder2.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), param1.get());

  // set kernel for SyncBNGrad
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder1.SetOutputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder1.SetInputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  builder1.SetOutputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  builder1.SetKernelType(TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), bn_grad.get());
  // do sync_bn_grad_split pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::SyncBnGradSplit>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_sync_bn_grad_split", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
