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
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "pipeline/jit/resource.h"
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "backend/common/optimizer/optimizer.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/ir_fission/bn_split.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWBnSplit : public BackendCommon {
 public:
  TestHWBnSplit() : get_py_fun_("gtest_input.pre_activate.bn_split", true) {}
  ~TestHWBnSplit() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWBnSplit, test_bn_split_tbe) {
  /*
   * def before(x, w, scale, b, mean, variance):
   *     sum = add(x, w)
   *     bn_output = bn(sum, scale, b, mean, variance)
   *     item0 = tuple_getitem(bn_output, 0)
   *     return item0
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_bn_split_tbe", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  std::vector<int64_t> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, b_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);

  // get kernel
  auto ret = kernel_graph->get_return();
  EXPECT_NE(ret, nullptr);
  EXPECT_TRUE(ret->inputs().size() == 2);
  auto make_tuple = ret->input(1)->cast<CNodePtr>();
  EXPECT_NE(make_tuple, nullptr);
  EXPECT_TRUE(make_tuple->inputs().size() == 2);
  auto item0 = make_tuple->input(1)->cast<CNodePtr>();
  EXPECT_NE(item0, nullptr);
  EXPECT_TRUE(item0->inputs().size() == 3);
  auto bn = item0->input(1);
  EXPECT_NE(bn, nullptr);
  EXPECT_TRUE(bn->isa<CNode>());

  // set kernel for BN
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder.SetOutputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder.SetInputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  builder.SetOutputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), bn.get());

  // do bn_split_pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::BnSplit>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_bn_split_tbe", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWBnSplit, test_sync_bn_split_tbe) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_sync_bn_split_tbe", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  std::vector<int64_t> shp_b{64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_b);
  AbstractBasePtrList args_spec_list{x_abstract, b_abstract, b_abstract, b_abstract, b_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);

  // get kernel
  auto ret = kernel_graph->get_return();
  EXPECT_NE(ret, nullptr);
  EXPECT_TRUE(ret->inputs().size() == 2);
  auto make_tuple = ret->input(1)->cast<CNodePtr>();
  EXPECT_NE(make_tuple, nullptr);
  EXPECT_TRUE(make_tuple->inputs().size() == 2);
  auto item0 = make_tuple->input(1)->cast<CNodePtr>();
  EXPECT_NE(item0, nullptr);
  EXPECT_TRUE(item0->inputs().size() == 3);
  auto bn = item0->input(1);
  EXPECT_NE(bn, nullptr);
  EXPECT_TRUE(bn->isa<CNode>());

  // set kernel for SyncBN
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder.SetOutputsFormat(
    {kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
  builder.SetInputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  builder.SetOutputsDeviceType(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), bn.get());

  // do sync_bn_split_pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::SyncBnSplit>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_sync_bn_split_tbe", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
