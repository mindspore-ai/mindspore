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
#include "frontend/operator/ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass_manager.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_info.h"
#include "utils/ms_context.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/format_type/insert_trans_op.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestHWInsertTransOp : public BackendCommon {
 public:
  TestHWInsertTransOp() : getPyFun_("gtest_input.pre_activate.insert_trans_op_test", true) {}
  ~TestHWInsertTransOp() override = default;
  FuncGraphPtr GetSingleOutputGraph(std::string func_name, std::string sub_func_name, std::string format) {
    FuncGraphPtr g = getPyFun_.CallAndParseRet(func_name, sub_func_name);
    std::vector<int64_t> shp{2, 32, 224, 224};
    auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
    AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
    auto fg = GetKernelGraph(g, args_spec_list);
    auto ret = fg->get_return();
    EXPECT_NE(ret->input(1), nullptr);
    EXPECT_NE(ret->input(1)->cast<CNodePtr>()->input(1), nullptr);
    auto add = ret->input(1)->cast<CNodePtr>()->input(1);
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({format, format});
    builder.SetInputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
    builder.SetInputsReshapeType({"", ""});
    builder.SetOutputsReshapeType({""});
    builder.SetOutputsFormat({format});
    builder.SetOutputsDeviceType({kFloat16->type_id()});
    builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    add->set_kernel_info(std::make_shared<device::KernelInfo>());
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), add.get());
    return fg;
  }
  FuncGraphPtr GetMutilpleOutputGraph(std::string func_name, std::string sub_func_name, std::string format) {
    FuncGraphPtr g = getPyFun_.CallAndParseRet(func_name, sub_func_name);
    std::vector<int64_t> shp{2, 32, 224, 224};
    auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
    AbstractBasePtrList args_spec_list{x_abstract};
    auto fg = GetKernelGraph(g, args_spec_list);

    // Set selectedKernelInfo for max_pool
    auto ret = fg->get_return();
    EXPECT_NE(ret->input(1), nullptr);
    EXPECT_NE(ret->input(1)->cast<CNodePtr>()->input(1), nullptr);
    EXPECT_NE(ret->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>()->input(1), nullptr);
    auto max_pool = ret->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>()->input(1);
    KernelBuildInfoBuilder builder;
    builder.SetInputsReshapeType({""});
    builder.SetOutputsReshapeType({"", ""});
    builder.SetInputsFormat({kOpFormat_DEFAULT});
    builder.SetInputsDeviceType({kFloat16->type_id()});
    builder.SetOutputsFormat({format, format});
    builder.SetOutputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
    builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
    max_pool->set_kernel_info(std::make_shared<device::KernelInfo>());
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), max_pool.get());
    return fg;
  }
  UT::PyFuncGraphFetcher getPyFun_;
};

class MockInsertTransOpKernelSelectTrans4Dto5D : public KernelSelect {
 public:
  bool is_four_to_five;
  MockInsertTransOpKernelSelectTrans4Dto5D() = default;
  ~MockInsertTransOpKernelSelectTrans4Dto5D() override = default;
  void SelectKernel(const CNodePtr &cnode) override {
    KernelBuildInfoBuilder builder;
    builder.SetInputsReshapeType({""});
    builder.SetOutputsReshapeType({""});
    builder.SetInputsFormat({"NCHW"});
    builder.SetInputsDeviceType({kFloat16->type_id()});
    builder.SetOutputsFormat({"NC1HWC0"});
    builder.SetOutputsDeviceType({kFloat16->type_id()});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
  }
};

TEST_F(TestHWInsertTransOp, test_insert_trans_op_for_single_output) {
  /*
   * def test_insert_trans_op_for_single_output(x, y):
   *     res = add(x, y)
   *     output = make_tuple(res)
   *     return output
   *
   */
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  auto fg = GetSingleOutputGraph("test_insert_trans_op_for_single_output", "before", "NC1HWC0");
  // Do insert_trans_op_ pass of hardware opt
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockInsertTransOpKernelSelectTrans4Dto5D>();
  pass_manager->AddPass(insert_trans_op_pass);
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g = graph_optimizer->Optimize(fg);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_insert_trans_op_for_single_output", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_g));
}

TEST_F(TestHWInsertTransOp, test_insert_trans_op_for_multiple_output) {
  /*
   * def test_insert_trans_op_for_multiple_output():
   *     max_pool_res = max_pool(x)
   *     res = tuple_getitem(max_pool_res, 0)
   *     output = make_tuple(res)
   *     return output
   */
  FuncGraphPtr fg = GetMutilpleOutputGraph("test_insert_trans_op_for_multiple_output", "before", "NC1HWC0");
  // Do insert_trans_op_ pass of hardware opt
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockInsertTransOpKernelSelectTrans4Dto5D>();
  pass_manager->AddPass(insert_trans_op_pass);
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g = graph_optimizer->Optimize(fg);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_insert_trans_op_for_multiple_output", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_g));
}
}  // namespace opt
}  // namespace mindspore
