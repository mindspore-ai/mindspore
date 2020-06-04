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
#include "ir/meta_tensor.h"
#include "ir/manager.h"
#include "debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "session/anf_runtime_algorithm.h"
#include "pre_activate/common/optimizer.h"
#include "pre_activate/common/pass_manager.h"
#include "device/kernel_info.h"
#include "pre_activate/ascend/format_type/insert_cast.h"
#include "kernel/kernel_build_info.h"
#include "utils/utils.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace opt {
class TestHWInsertCast : public BackendCommon {
 public:
  TestHWInsertCast() : getPyFun_("gtest_input.pre_activate.mixed_precision_test", true) {}
  ~TestHWInsertCast() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestHWInsertCast, test_insert_cast_op_for_single_output) {
  /*
   * def test_insert_cast_op_for_single_output(x, y):
   *     res = add((x, y))
   *     return res
   */
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_insert_cast_op_for_single_output", "before");
  std::vector<int> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(func_graph, nullptr);

  // Set selectedKernelInfo
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::AKG_KERNEL);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NC1HWC0"});
  builder1.SetInputsDeviceType({kFloat32->type_id()});
  builder1.SetOutputsFormat({"NC1HWC0"});
  builder1.SetOutputsDeviceType({kFloat32->type_id()});
  builder1.SetFusionType(kernel::FusionType::ELEMWISE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  builder1.SetKernelType(KernelType::AKG_KERNEL);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto& node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if (node->isa<Parameter>()) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), node.get());
    } else if (node != func_graph->get_return() && AnfAlgo::IsRealKernel(node)) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
    }
  }
  // Do insert cast pass of hardware opt
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertCast>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  EXPECT_NE(new_graph, nullptr);

  // check result
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_insert_cast_op_for_single_output", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWInsertCast, test_insert_cast_op_for_multiple_output) {
  /*
   * def test_insert_cast_op_for_multiple_output():
   *     output = max_pool(w)
   *     res = tuple_getitem(output, 0)
   *     return res
   */
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_insert_cast_op_for_multiple_output", "before");
  std::vector<int> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(func_graph, nullptr);

  // Set selectedKernelInfo
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"DefaultFormat"});
  builder.SetOutputsFormat({"NC1HWC0", "NC1HWC0"});
  builder.SetInputsDeviceType({kFloat16->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id(), kFloat16->type_id()});
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NC1HWC0"});
  builder1.SetInputsDeviceType({kFloat32->type_id()});
  builder1.SetOutputsFormat({"DefaultFormat"});
  builder1.SetOutputsDeviceType({kFloat32->type_id()});
  builder1.SetFusionType(kernel::FusionType::ELEMWISE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  builder1.SetKernelType(KernelType::AKG_KERNEL);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto& node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if (node->isa<Parameter>()) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), node.get());
    } else if (node != func_graph->get_return() && AnfAlgo::IsRealKernel(node)) {
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
    }
  }

  // Do insert cast pass of hardware opt
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertCast>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(func_graph);
  EXPECT_NE(new_graph, nullptr);

  // check result
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_insert_cast_op_for_multiple_output", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
