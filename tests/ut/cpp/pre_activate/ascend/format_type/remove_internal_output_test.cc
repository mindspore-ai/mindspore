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
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "plugin/device/ascend/optimizer/format_type/remove_internal_output.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/format_type/insert_trans_op.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestHWRemoveInternalOutput : public BackendCommon {
 public:
  TestHWRemoveInternalOutput() : getPyFun_("gtest_input.pre_activate.remove_internal_output_test", true) {}
  ~TestHWRemoveInternalOutput() override = default;

  AnfNodePtr GetMakeTuple(const KernelGraphPtr &kg) {
    auto ret = kg->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    auto make_tuple = ret->input(1);
    return make_tuple;
  }

  KernelGraphPtr GetSingleOutputGraph(const std::string &func_name, const std::string &sub_func_name) {
    FuncGraphPtr g = getPyFun_.CallAndParseRet(func_name, sub_func_name);
    std::vector<int64_t> shp{2, 32, 224, 224};
    auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
    AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
    auto kg = GetKernelGraph(g, args_spec_list);
    auto make_tuple = GetMakeTuple(kg);
    auto add = make_tuple->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(add);
    kg->AddInternalOutput(add, add, 0, true);
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
    builder.SetInputsReshapeType({"", ""});
    builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
    builder.SetOutputsReshapeType({""});
    builder.SetOutputsFormat({kOpFormat_NC1HWC0});
    builder.SetOutputsDeviceType({kFloat16->type_id()});
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    add->set_kernel_info(std::make_shared<device::KernelInfo>());
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), add.get());
    return kg;
  }

  KernelGraphPtr GetMutilpleOutputGraph(const std::string &func_name, const std::string &sub_func_name) {
    FuncGraphPtr g = getPyFun_.CallAndParseRet(func_name, sub_func_name);
    std::vector<int64_t> shp{2, 32, 224, 224};
    auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
    AbstractBasePtrList args_spec_list{x_abstract};
    auto kg = GetKernelGraph(g, args_spec_list);
    auto output_make_tuple = GetMakeTuple(kg);
    auto make_tuple = output_make_tuple->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(make_tuple);
    auto tuple_getitem1 = make_tuple->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(tuple_getitem1);
    auto tuple_getitem2 = make_tuple->cast<CNodePtr>()->input(2);
    MS_EXCEPTION_IF_NULL(tuple_getitem2);
    auto max_pool = tuple_getitem1->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(max_pool);
    kg->AddInternalOutput(tuple_getitem1, max_pool, 0, true);
    kg->AddInternalOutput(tuple_getitem2, max_pool, 1, true);
    KernelBuildInfoBuilder builder;
    builder.SetInputsReshapeType({""});
    builder.SetOutputsReshapeType({"", ""});
    builder.SetInputsFormat({kOpFormat_DEFAULT});
    builder.SetInputsDeviceType({kFloat32->type_id()});
    builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    builder.SetOutputsFormat({kOpFormat_NC1HWC0, kOpFormat_NC1HWC0});
    builder.SetOutputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
    max_pool->set_kernel_info(std::make_shared<device::KernelInfo>());
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), max_pool.get());
    return kg;
  }
  UT::PyFuncGraphFetcher getPyFun_;
};

class MockRemoveInternalOutputTransOpKernelSelect : public KernelSelect {
 public:
  MockRemoveInternalOutputTransOpKernelSelect() = default;
  ~MockRemoveInternalOutputTransOpKernelSelect() override = default;
  void SelectKernel(const CNodePtr &cnode) override {
    KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_NC1HWC0});
    builder.SetInputsDeviceType({kFloat16->type_id()});
    builder.SetOutputsFormat({kOpFormat_DEFAULT});
    builder.SetOutputsDeviceType({kFloat32->type_id()});
    builder.SetInputsReshapeType({""});
    builder.SetOutputsReshapeType({""});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
  }
};

TEST_F(TestHWRemoveInternalOutput, test_remove_internal_output_trans_op_for_single_output) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  auto kg = GetSingleOutputGraph("test_remove_internal_output_trans_op_for_single_output", "before");
  // insert trans op for output
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockRemoveInternalOutputTransOpKernelSelect>();
  pass_manager->AddPass(insert_trans_op_pass);
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g = graph_optimizer->Optimize(kg);
  FuncGraphPtr g_after =
    getPyFun_.CallAndParseRet("test_remove_internal_output_trans_op_for_single_output", "after_insert_trans_op");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_g));

  auto make_tuple = GetMakeTuple(kg);
  auto trans_data = make_tuple->cast<CNodePtr>()->input(1);
  EXPECT_TRUE(kg->IsInternalOutput(trans_data, 0));

  // remove trans op for internal output
  auto graph_optimizer1 = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager1 = std::make_shared<opt::PassManager>();
  auto remove_internal_output_trans_op_pass = std::make_shared<opt::RemoveInternalOutputTransOp>();
  pass_manager1->AddPass(remove_internal_output_trans_op_pass);
  graph_optimizer1->AddPassManager(pass_manager1);
  auto new_g1 = graph_optimizer1->Optimize(new_g);
  FuncGraphPtr g_after1 = getPyFun_.CallAndParseRet("test_remove_internal_output_trans_op_for_single_output",
                                                    "after_remove_internal_output_trans_op");
  EXPECT_TRUE(CheckEqualGraph(g_after1, new_g1));
}

TEST_F(TestHWRemoveInternalOutput, test_remove_internal_output_trans_op_for_multiple_output) {
  auto kg = GetMutilpleOutputGraph("test_remove_internal_output_trans_op_for_multiple_output", "before");
  // insert trans op for output
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockRemoveInternalOutputTransOpKernelSelect>();
  pass_manager->AddPass(insert_trans_op_pass);
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g = graph_optimizer->Optimize(kg);
  FuncGraphPtr g_after =
    getPyFun_.CallAndParseRet("test_remove_internal_output_trans_op_for_multiple_output", "after_insert_trans_op");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_g));

  auto output_make_tuple = GetMakeTuple(kg);
  auto make_tuple = output_make_tuple->cast<CNodePtr>()->input(1);
  auto tuple_getitem = make_tuple->cast<CNodePtr>()->input(1);
  auto make_tuple1 = tuple_getitem->cast<CNodePtr>()->input(1);
  auto trans_data1 = make_tuple1->cast<CNodePtr>()->input(1);
  auto trans_data2 = make_tuple1->cast<CNodePtr>()->input(2);
  EXPECT_TRUE(kg->IsInternalOutput(trans_data1, 0));
  EXPECT_TRUE(kg->IsInternalOutput(trans_data2, 0));

  // remove trans op for internal output
  auto graph_optimizer1 = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager1 = std::make_shared<opt::PassManager>();
  auto remove_internal_output_trans_op_pass = std::make_shared<opt::RemoveInternalOutputTransOp>();
  pass_manager1->AddPass(remove_internal_output_trans_op_pass);
  graph_optimizer1->AddPassManager(pass_manager1);
  auto new_g1 = graph_optimizer1->Optimize(new_g);
  FuncGraphPtr g_after1 = getPyFun_.CallAndParseRet("test_remove_internal_output_trans_op_for_multiple_output",
                                                    "after_remove_internal_output_trans_op");
  EXPECT_TRUE(CheckEqualGraph(g_after1, new_g1));
}
}  // namespace opt
}  // namespace mindspore
