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
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "backend/common/pass/communication_op_fusion.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/kernel_info.h"
#include "include/backend/optimizer/pass_manager.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
class TestHWAllReduceFusion : public BackendCommon {
 public:
  TestHWAllReduceFusion() : getPyFun_("gtest_input.pre_activate.ir_fusion_test", true) {}
  ~TestHWAllReduceFusion() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestHWAllReduceFusion, test_fusion_all) {
  getPyFun_.SetDoResolve(true);
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_all_reduce_fusion_all", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(func_graph, nullptr);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::AKG_KERNEL);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if ((node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kAllReduceOpName) || node->isa<Parameter>()) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
    }
  }
  // do all reduce fusion
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  EXPECT_NE(new_graph, nullptr);
  // check result
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_all_reduce_fusion_all", "after");
  EXPECT_NE(g_after, nullptr);
  EXPECT_TRUE(CheckEqualGraph(new_graph, g_after));
}

TEST_F(TestHWAllReduceFusion, test_fusion_group) {
  getPyFun_.SetDoResolve(true);
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_all_reduce_fusion_group", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(func_graph, nullptr);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::AKG_KERNEL);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if ((node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kAllReduceOpName) || node->isa<Parameter>()) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
    }
  }
  // do all reduce fusion
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AllReduceFusion>(2));
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  EXPECT_NE(new_graph, nullptr);
  // check result
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_all_reduce_fusion_group", "after1");
  EXPECT_NE(g_after, nullptr);
  EXPECT_TRUE(CheckEqualGraph(new_graph, g_after));
}

TEST_F(TestHWAllReduceFusion, test_fusion_op) {
  getPyFun_.SetDoResolve(true);
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_all_reduce_fusion_group", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(func_graph, nullptr);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::AKG_KERNEL);
  auto node_list = TopoSort(func_graph->get_return());
  int count = 0;
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if ((node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kAllReduceOpName) || node->isa<Parameter>()) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
    }
    if (node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kAllReduceOpName) {
      if (count == 0) {
        common::AnfAlgo::SetNodeAttr("op", MakeValue("max"), node);
        count = 1;
      } else {
        common::AnfAlgo::SetNodeAttr("op", MakeValue("sum"), node);
        count = 0;
      }
    }
  }
  // do all reduce fusion
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  EXPECT_NE(new_graph, nullptr);
  // check result
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_all_reduce_fusion_group", "after2");
  EXPECT_NE(g_after, nullptr);
  EXPECT_TRUE(CheckEqualGraph(new_graph, g_after));
}

TEST_F(TestHWAllReduceFusion, test_fusion_sorted) {
  getPyFun_.SetDoResolve(true);
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_all_reduce_fusion_all", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(func_graph, nullptr);
  auto ret = func_graph->get_return();
  auto make_tuple = ret->input(1);
  auto make_tuple1 = make_tuple->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  for (size_t i = 1; i < make_tuple1->inputs().size(); ++i) {
    common::AnfAlgo::SetNodeAttr(kAttrIndex, MakeValue(SizeToLong(i)), make_tuple1->input(i));
  }
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::AKG_KERNEL);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if ((node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kAllReduceOpName) || node->isa<Parameter>()) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
    }
  }
  // do all reduce fusion
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  EXPECT_NE(new_graph, nullptr);
  // check result
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_all_reduce_fusion_all", "after");
  EXPECT_NE(g_after, nullptr);
  EXPECT_TRUE(CheckEqualGraph(new_graph, g_after));
}
}  // namespace opt
}  // namespace mindspore
