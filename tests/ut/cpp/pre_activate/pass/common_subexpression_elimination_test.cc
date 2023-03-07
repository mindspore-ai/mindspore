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
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/optimizer/pass_manager.h"
#include "include/backend/kernel_info.h"
#include "backend/common/pass/common_subexpression_elimination.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
class TestHWCSE : public BackendCommon {
 public:
  TestHWCSE() : getPyFun_("gtest_input.pre_activate.hw_opt_test", true) {}
  ~TestHWCSE() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

kernel::KernelBuildInfoPtr CreateKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                 const std::vector<std::string> &outputs_format,
                                                 const std::vector<TypeId> &inputs_device_type,
                                                 const std::vector<TypeId> &outputs_device_type,
                                                 KernelType kernel_type) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat(inputs_format);
  builder1.SetOutputsFormat(outputs_format);
  builder1.SetInputsDeviceType(inputs_device_type);
  builder1.SetOutputsDeviceType(outputs_device_type);
  builder1.SetKernelType(kernel_type);
  return builder1.Build();
}

TEST_F(TestHWCSE, test_func_graph_cse) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_func_graph_cse", "g1");
  std::vector<int64_t> shp_x{32, 3, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  FuncGraphManagerPtr manager = Manage(func_graph);
  ASSERT_TRUE(manager != nullptr);
  ASSERT_EQ(manager->all_nodes().size(), 10);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::CommonSubexpressionElimination>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 8);

  g = getPyFun_.CallAndParseRet("test_func_graph_cse", "g2");
  func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  manager = Manage(func_graph);
  ASSERT_TRUE(manager != nullptr);
  ASSERT_EQ(manager->all_nodes().size(), 22);
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 14);
}

TEST_F(TestHWCSE, test_func_graph_cse_with_null_kernel_info) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_func_graph_cse", "g1");
  std::vector<int64_t> shp_x{32, 3, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  FuncGraphManagerPtr manager = Manage(func_graph);
  ASSERT_TRUE(manager != nullptr);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  auto ret = func_graph->get_return();
  EXPECT_NE(ret, nullptr);
  auto mul = ret->input(1);
  EXPECT_NE(mul, nullptr);
  auto add1 = mul->cast<CNodePtr>()->input(1);
  auto add2 = mul->cast<CNodePtr>()->input(2);
  EXPECT_NE(add1, nullptr);
  EXPECT_NE(add2, nullptr);
  add1->set_kernel_info(std::make_shared<device::KernelInfo>());
  add2->set_kernel_info(std::make_shared<device::KernelInfo>());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::CommonSubexpressionElimination>());
  optimizer->AddPassManager(pm);
  // one of the kernel info is null
  add1->set_kernel_info(nullptr);
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  // one of the kernel build info is null
  add1->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat16}, {kNumberTypeFloat32}, TBE_KERNEL),
    add2.get());
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  // both kernel build info is null
  AnfAlgo::SetSelectKernelBuildInfo(nullptr, add2.get());
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 8);
}

TEST_F(TestHWCSE, test_func_graph_cse_with_diff_kernel_info) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_func_graph_cse", "g1");
  std::vector<int64_t> shp_x{32, 3, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  FuncGraphManagerPtr manager = Manage(func_graph);
  ASSERT_TRUE(manager != nullptr);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  auto ret = func_graph->get_return();
  EXPECT_NE(ret, nullptr);
  auto mul = ret->input(1);
  EXPECT_NE(mul, nullptr);
  auto add1 = mul->cast<CNodePtr>()->input(1);
  auto add2 = mul->cast<CNodePtr>()->input(2);
  EXPECT_NE(add1, nullptr);
  EXPECT_NE(add2, nullptr);
  add1->set_kernel_info(std::make_shared<device::KernelInfo>());
  add2->set_kernel_info(std::make_shared<device::KernelInfo>());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::CommonSubexpressionElimination>());
  optimizer->AddPassManager(pm);
  // Different data type
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat32}, {kNumberTypeFloat32}, TBE_KERNEL),
    add1.get());
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat16}, {kNumberTypeFloat32}, TBE_KERNEL),
    add2.get());
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  // Different format
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat32}, {kNumberTypeFloat32}, TBE_KERNEL),
    add1.get());
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_NC1HWC0}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat32}, {kNumberTypeFloat32}, TBE_KERNEL),
    add2.get());
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  // Different kernel type
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat32}, {kNumberTypeFloat32}, TBE_KERNEL),
    add1.get());
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_NC1HWC0}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat32}, {kNumberTypeFloat32}, UNKNOWN_KERNEL_TYPE),
    add2.get());
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 10);
  // same kernel build info
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat32}, {kNumberTypeFloat32}, TBE_KERNEL),
    add1.get());
  AnfAlgo::SetSelectKernelBuildInfo(
    CreateKernelBuildInfo({kOpFormat_DEFAULT, kOpFormat_DEFAULT}, {kOpFormat_DEFAULT},
                          {kNumberTypeFloat32, kNumberTypeFloat32}, {kNumberTypeFloat32}, TBE_KERNEL),
    add2.get());
  optimizer->Optimize(func_graph);
  ASSERT_EQ(manager->all_nodes().size(), 8);
}
}  // namespace opt
}  // namespace mindspore
