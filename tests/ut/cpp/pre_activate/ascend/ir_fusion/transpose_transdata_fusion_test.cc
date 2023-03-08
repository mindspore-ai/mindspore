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
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "kernel/oplib/oplib.h"
#include "utils/ms_context.h"
#include "include/common/utils/anfalgo.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/format_type/insert_trans_op.h"
#include "plugin/device/ascend/optimizer/ir_fusion/transpose_transdata_fusion.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
class TestHWTransposeTransdataFusion : public BackendCommon {
 public:
  TestHWTransposeTransdataFusion() : get_py_fun_("gtest_input.pre_activate.transpose_transdata_fusion_test", true) {}
  ~TestHWTransposeTransdataFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWTransposeTransdataFusion, test_transpose_transdata_fusion) {
  /*
   * def before(input0, input1):
   * transpose = transpose(input0, input1)
   * transdata = Transdata(transpose)
   * return transdata
   */
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_transpose_transdata_fusion", "before");
  std::vector<int64_t> shp{2, 4, 8, 16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto temp = ret->input(1)->cast<CNodePtr>();
  auto transpose = temp->input(1)->cast<CNodePtr>();
  EXPECT_NE(transpose, nullptr);

  KernelBuildInfoBuilder builder;
  builder.SetInputsReshapeType({""});
  builder.SetOutputsReshapeType({""});
  builder.SetInputsFormat({"DefaultFormat"});
  builder.SetInputsDeviceType({kFloat16->type_id()});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->set_select_kernel_build_info(builder.Build());
  transpose->set_kernel_info(kernel_info);

  // insert transdata
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  pm->AddPass(insert_trans_op_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  // modify transdata
  auto nodes = TopoSort(new_graph->get_return());
  for (auto &node : nodes) {
    if (AnfUtils::IsRealCNodeKernel(node) && common::AnfAlgo::GetCNodeName(node) == "TransData") {
      KernelBuildInfoBuilder builder2;
      builder2.SetInputsFormat({"DefaultFormat"});
      builder2.SetInputsDeviceType({kFloat16->type_id()});
      builder2.SetOutputsFormat({"NC1HWC0"});
      builder2.SetOutputsDeviceType({kFloat16->type_id()});
      builder2.SetInputsReshapeType({""});
      builder2.SetOutputsReshapeType({""});
      builder2.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      builder2.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      builder2.SetKernelType(KernelType::TBE_KERNEL);
      builder2.SetProcessor(kernel::Processor::AICORE);
      AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), node.get());
    }
  }

  // transpose transdata fusion
  auto optimizer2 = std::make_shared<opt::GraphOptimizer>();
  auto pm2 = std::make_shared<opt::PassManager>();
  auto transpose_transdata_pass = std::make_shared<opt::TransposeTransDataFusion>();
  pm2->AddPass(transpose_transdata_pass);
  optimizer2->AddPassManager(pm2);
  new_graph = optimizer2->Optimize(new_graph);

  // check
  ret = new_graph->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  temp = ret->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "input(1) name:" << temp->fullname_with_scope();
  EXPECT_NE(temp->input(1), nullptr);
  auto temp_node = temp->input(1)->cast<CNodePtr>();
  MS_LOG(INFO) << "input(11) name:" << temp_node->fullname_with_scope();

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_transpose_transdata_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
