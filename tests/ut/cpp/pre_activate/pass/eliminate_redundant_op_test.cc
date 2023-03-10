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
#include "kernel/kernel.h"
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
// #include "runtime/device/optimizer/pass/insert_trans_op.h"
#include "plugin/device/ascend/optimizer/format_type/insert_cast.h"
#include "backend/common/pass/eliminate_redundant_op.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/kernel_info.h"
#include "utils/ms_context.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/format_type/insert_trans_op.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestHWEliminateRedundantOp : public BackendCommon {
 public:
  TestHWEliminateRedundantOp() : getPyFun_("gtest_input.pre_activate.eliminate_redundant_op_test", true) {}
  ~TestHWEliminateRedundantOp() override = default;

  UT::PyFuncGraphFetcher getPyFun_;
};

class MockEliminate5To4And4To5KernelSelect : public KernelSelect {
 public:
  MockEliminate5To4And4To5KernelSelect() = default;
  ~MockEliminate5To4And4To5KernelSelect() override = default;
  void SelectKernel(const CNodePtr &cnode) override {
    KernelBuildInfoBuilder builder;
    builder.SetInputsReshapeType({""});
    builder.SetOutputsReshapeType({""});
    builder.SetInputsFormat({"NCHW"});
    builder.SetInputsDeviceType({kFloat16->type_id()});
    builder.SetOutputsFormat({"NC1HWC0"});
    builder.SetOutputsDeviceType({kFloat16->type_id()});
    builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
  }
};

TEST_F(TestHWEliminateRedundantOp, test_eliminate_5to4_4to5) {
  /*
   * def test_eliminate_5to4_4to5(x, y):
   *     sum = add(x, y)
   *     res = sub(sum, y)
   *     output = make_tuple(res)
   *     return output
   */
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_eliminate_5to4_4to5", "before");
  // Renormalize func_graph to infer and set shape and type information.
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  // Set selectedKernelInfo for add, sub
  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>()->input(1), nullptr);

  auto tuple = ret->input(1)->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple, nullptr);
  EXPECT_NE(tuple->cast<CNodePtr>()->input(1), nullptr);

  auto sub = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(sub, nullptr);
  EXPECT_NE(sub->cast<CNodePtr>()->input(1), nullptr);
  auto add = sub->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetInputsReshapeType({"", ""});
  builder.SetOutputsReshapeType({""});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  sub->set_kernel_info(std::make_shared<device::KernelInfo>());
  add->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), sub.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), add.get());

  // Do insert_trans_op_ pass of hardware opt
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockEliminate5To4And4To5KernelSelect>();
  pass_manager->AddPass(insert_trans_op_pass);
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g1 = graph_optimizer->Optimize(kg);
  EXPECT_NE(new_g1, nullptr);
  FuncGraphPtr g_after1 = getPyFun_.CallAndParseRet("test_eliminate_5to4_4to5", "after1");
  EXPECT_NE(g_after1, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after1, new_g1));

  // Do eliminate_5to4_4to5_ pass of hardware opt
  auto graph_optimizer2 = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager2 = std::make_shared<opt::PassManager>();
  pass_manager2->AddPass(std::make_shared<opt::EliminateRedundantOp>());
  graph_optimizer2->AddPassManager(pass_manager2);
  auto new_g2 = graph_optimizer2->Optimize(new_g1);
  EXPECT_NE(new_g2, nullptr);
  FuncGraphPtr g_after2 = getPyFun_.CallAndParseRet("test_eliminate_5to4_4to5", "after2");
  EXPECT_NE(g_after2, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after2, new_g2));
}

TEST_F(TestHWEliminateRedundantOp, test_eliminate_cast) {
  /*
   * def test_eliminate_cast(x, y):
   *     sum = add(x, y)
   *     res = sub(sum, y)
   *     output = make_tuple(res)
   *     return output
   */
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_eliminate_cast", "before");
  // Renormalize func_graph to infer and set shape and type information.
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  // Set selectedKernelInfo for add, sub
  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>()->input(1), nullptr);

  auto tuple = ret->input(1)->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple, nullptr);
  EXPECT_NE(tuple->cast<CNodePtr>()->input(1), nullptr);

  auto sub = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(sub, nullptr);
  EXPECT_NE(sub->cast<CNodePtr>()->input(1), nullptr);
  auto add = sub->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetInputsReshapeType({"", ""});
  builder.SetOutputsReshapeType({"", ""});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
  sub->set_kernel_info(std::make_shared<device::KernelInfo>());
  add->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), sub.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), add.get());

  // Do insert_trans_op_ pass of hardware opt
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  pass_manager->AddPass(std::make_shared<opt::InsertCast>());
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g1 = graph_optimizer->Optimize(kg);
  EXPECT_NE(new_g1, nullptr);
  FuncGraphPtr g_after1 = getPyFun_.CallAndParseRet("test_eliminate_cast", "after1");
  EXPECT_NE(g_after1, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after1, new_g1));

  // Do eliminate_5to4_4to5_ pass of hardware opt
  auto graph_optimizer2 = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager2 = std::make_shared<opt::PassManager>();
  pass_manager2->AddPass(std::make_shared<opt::EliminateRedundantOp>());
  graph_optimizer2->AddPassManager(pass_manager2);
  auto new_g2 = graph_optimizer2->Optimize(new_g1);
  EXPECT_NE(new_g2, nullptr);
  FuncGraphPtr g_after2 = getPyFun_.CallAndParseRet("test_eliminate_cast", "after2");
  EXPECT_NE(g_after2, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after2, new_g2));
}

TEST_F(TestHWEliminateRedundantOp, test_eliminate_cast_depend_cast) {
  /*
   * def test_eliminate_cast_depend_cast(x, y):
   *     sum = add(x, y)
   *     sum_depend = depend(sum, x)
   *     res = sub(sum_depend, y)
   *     output = make_tuple(res)
   *     return output
   */
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_eliminate_cast_depend_cast", "before");
  // Renormalize func_graph to infer and set shape and type information.
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  // Set selectedKernelInfo for add, sub
  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>()->input(1), nullptr);

  auto tuple = ret->input(1)->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple, nullptr);
  EXPECT_NE(tuple->cast<CNodePtr>()->input(1), nullptr);

  auto sub = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(sub, nullptr);
  EXPECT_NE(sub->cast<CNodePtr>()->input(1), nullptr);
  auto depend = sub->cast<CNodePtr>()->input(1);
  EXPECT_NE(depend, nullptr);
  EXPECT_NE(depend->cast<CNodePtr>()->input(1), nullptr);

  auto depend2 = depend->cast<CNodePtr>()->input(1);
  EXPECT_NE(depend2, nullptr);
  EXPECT_NE(depend2->cast<CNodePtr>()->input(1), nullptr);

  auto depend3 = depend2->cast<CNodePtr>()->input(1);
  EXPECT_NE(depend3, nullptr);
  EXPECT_NE(depend3->cast<CNodePtr>()->input(1), nullptr);
  auto add = depend3->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetInputsReshapeType({"", ""});
  builder.SetOutputsReshapeType({""});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR, kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  sub->set_kernel_info(std::make_shared<device::KernelInfo>());
  add->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), sub.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), add.get());

  // Do insert_trans_op_ pass of hardware opt
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  pass_manager->AddPass(std::make_shared<opt::InsertCast>());
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g1 = graph_optimizer->Optimize(kg);
  EXPECT_NE(new_g1, nullptr);
  FuncGraphPtr g_after1 = getPyFun_.CallAndParseRet("test_eliminate_cast_depend_cast", "after1");
  EXPECT_NE(g_after1, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after1, new_g1));

  // Do eliminate_5to4_4to5_ pass of hardware opt
  auto graph_optimizer2 = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager2 = std::make_shared<opt::PassManager>();
  pass_manager2->AddPass(std::make_shared<opt::EliminateRedundantOp>());
  graph_optimizer2->AddPassManager(pass_manager2);
  auto new_g2 = graph_optimizer2->Optimize(new_g1);
  EXPECT_NE(new_g2, nullptr);
  FuncGraphPtr g_after2 = getPyFun_.CallAndParseRet("test_eliminate_cast_depend_cast", "after2");
  EXPECT_NE(g_after2, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after2, new_g2));
}

}  // namespace opt
}  // namespace mindspore
