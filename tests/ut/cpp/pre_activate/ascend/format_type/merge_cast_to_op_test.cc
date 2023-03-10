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
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/anfalgo.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/format_type/merge_cast_to_op.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWMergeCastToOp : public BackendCommon {
 public:
  TestHWMergeCastToOp() : get_py_fun_("gtest_input.pre_activate.merge_cast_to_op", true) {}
  ~TestHWMergeCastToOp() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

class MockMergeCastToOpKernelQuery : public KernelQuery {
 public:
  MockMergeCastToOpKernelQuery() = default;
  ~MockMergeCastToOpKernelQuery() override = default;
  void Query(const CNodePtr &kernel_node,
             std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) override {
    std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (op_name == kFour2FiveOpName) {
      kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({kOpFormat_NCHW});
      builder.SetOutputsFormat({kOpFormat_NC1HWC0});
      builder.SetInputsDeviceType({kNumberTypeFloat32});
      builder.SetOutputsDeviceType({kNumberTypeFloat16});
      kernel_info_list->push_back(builder.Build());
    } else if (op_name == kFive2FourOpName) {
      kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({kOpFormat_NC1HWC0});
      builder.SetOutputsFormat({kOpFormat_NCHW});
      builder.SetInputsDeviceType({kNumberTypeFloat16});
      builder.SetOutputsDeviceType({kNumberTypeFloat32});
      kernel_info_list->push_back(builder.Build());
    }
  }
};

TEST_F(TestHWMergeCastToOp, test_merge_cast_to_next_op) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_merge_cast_to_next_op", "before");
  ASSERT_NE(g, nullptr);

  // set abstract because four2five node cannot infer
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  g->parameters()[0]->set_abstract(x_abstract);
  g->get_return()->set_abstract(x_abstract);
  AnfNodePtr g_four2five = g->get_return()->input(1);
  g_four2five->set_abstract(x_abstract);
  AnfNodePtr g_cast = g_four2five->cast<CNodePtr>()->input(1);
  g_cast->set_abstract(x_abstract);

  // convert to kernel graph
  AbstractBasePtrList args_spec_list;
  auto kernel_graph = GetKernelGraph(g, args_spec_list, false);

  // get four2five
  auto ret = kernel_graph->get_return();
  EXPECT_NE(ret, nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_TRUE(ret->input(1)->isa<CNode>());
  auto make_tuple = ret->input(1)->cast<CNodePtr>();
  EXPECT_NE(make_tuple->input(1), nullptr);
  EXPECT_TRUE(make_tuple->input(1)->isa<CNode>());
  auto four2five = make_tuple->input(1)->cast<CNodePtr>();

  // set kernel for four2five
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_NCHW});
  builder1.SetOutputsFormat({kOpFormat_NC1HWC0});
  builder1.SetInputsDeviceType({kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), four2five.get());

  // get cast
  EXPECT_NE(four2five->input(1), nullptr);
  EXPECT_TRUE(four2five->input(1)->isa<CNode>());
  auto cast = four2five->input(1)->cast<CNodePtr>();

  // set kernel for cast
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({kOpFormat_NCHW});
  builder2.SetOutputsFormat({kOpFormat_NCHW});
  builder2.SetInputsDeviceType({kNumberTypeFloat32});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), cast.get());

  // do merge_cast_to_op_pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::MergeCastToOp>();
  pass->kernel_query_ = std::make_shared<MockMergeCastToOpKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_merge_cast_to_next_op", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWMergeCastToOp, test_merge_cast_to_prior_op) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_merge_cast_to_prior_op", "before");
  ASSERT_NE(g, nullptr);

  // set abstract because five2four node cannot infer
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  g->parameters()[0]->set_abstract(x_abstract);
  g->get_return()->set_abstract(x_abstract);
  AnfNodePtr g_cast = g->get_return()->input(1);
  g_cast->set_abstract(x_abstract);
  AnfNodePtr g_five2four = g_cast->cast<CNodePtr>()->input(1);
  g_five2four->set_abstract(x_abstract);

  // convert to kernel graph
  AbstractBasePtrList args_spec_list;
  auto kernel_graph = GetKernelGraph(g, args_spec_list, false);

  // get cast
  auto ret = kernel_graph->get_return();
  EXPECT_NE(ret, nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_TRUE(ret->input(1)->isa<CNode>());
  auto make_tuple = ret->input(1)->cast<CNodePtr>();
  EXPECT_NE(make_tuple->input(1), nullptr);
  EXPECT_TRUE(make_tuple->input(1)->isa<CNode>());
  auto cast = make_tuple->input(1)->cast<CNodePtr>();
  common::AnfAlgo::SetNodeAttr("dst_type", MakeValue("float32"), cast);

  // set kernel for cast
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_NCHW});
  builder1.SetOutputsFormat({kOpFormat_NCHW});
  builder1.SetInputsDeviceType({kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast.get());

  // get five2four
  EXPECT_NE(cast->input(1), nullptr);
  EXPECT_TRUE(cast->input(1)->isa<CNode>());
  auto five2four = cast->input(1)->cast<CNodePtr>();

  // set kernel for five2four
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({kOpFormat_NC1HWC0});
  builder2.SetOutputsFormat({kOpFormat_NCHW});
  builder2.SetInputsDeviceType({kNumberTypeFloat16});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), five2four.get());

  // do merge_cast_to_op_pass
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::MergeCastToOp>();
  pass->kernel_query_ = std::make_shared<MockMergeCastToOpKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_merge_cast_to_prior_op", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
