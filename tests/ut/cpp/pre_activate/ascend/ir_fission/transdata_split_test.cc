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
#include "runtime/device/kernel_info.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "kernel/oplib/oplib.h"
#include "include/common/debug/anf_ir_dump.h"
#include "utils/ms_context.h"
#include "include/common/utils/anfalgo.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/format_type/insert_trans_op.h"
#include "plugin/device/ascend/optimizer/ir_fission/transdata_split.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
class TestHWTransdataSplit : public BackendCommon {
 public:
  TestHWTransdataSplit() : get_py_fun_("gtest_input.pre_activate.transdata_split_test", true) {}
  ~TestHWTransdataSplit() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

class MockInsertTransOpKernelSelectTrans4Dto5D : public KernelSelect {
 public:
  MockInsertTransOpKernelSelectTrans4Dto5D() = default;
  ~MockInsertTransOpKernelSelectTrans4Dto5D() override = default;
  void SelectKernel(const CNodePtr &cnode) override {
    if (common::AnfAlgo::GetCNodeName(cnode) == "Four2Five") {
      KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({"NCHW"});
      builder.SetInputsDeviceType({kFloat16->type_id()});
      builder.SetOutputsFormat({"NC1HWC0"});
      builder.SetOutputsDeviceType({kFloat16->type_id()});
      builder.SetInputsReshapeType({""});
      builder.SetOutputsReshapeType({""});
      builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
    } else {
      KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({"NC1HWC0"});
      builder.SetInputsDeviceType({kFloat16->type_id()});
      builder.SetOutputsFormat({"NC1HWC0"});
      builder.SetOutputsDeviceType({kFloat16->type_id()});
      builder.SetInputsReshapeType({""});
      builder.SetOutputsReshapeType({""});
      builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
    }
  }
};

class MockTransdataSplitKernelSelect : public KernelSelect {
 public:
  MockTransdataSplitKernelSelect() = default;
  ~MockTransdataSplitKernelSelect() override = default;
  void SelectKernel(const CNodePtr &cnode) override {
    if (common::AnfAlgo::GetCNodeName(cnode) == kTransDataOpName) {
      KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({"NCHW"});
      builder.SetInputsDeviceType({kFloat16->type_id()});
      builder.SetOutputsFormat({"NCHW"});
      builder.SetOutputsDeviceType({kFloat16->type_id()});
      builder.SetInputsReshapeType({""});
      builder.SetOutputsReshapeType({""});
      builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
    } else {
      KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({"NCHW"});
      builder.SetInputsDeviceType({kFloat16->type_id()});
      builder.SetOutputsFormat({"NCHW"});
      builder.SetOutputsDeviceType({kFloat16->type_id()});
      builder.SetInputsReshapeType({""});
      builder.SetOutputsReshapeType({""});
      builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
    }
  }
};

TEST_F(TestHWTransdataSplit, test_transdata_split_fraz_nchw) {
  /*
   * def before(input0, input1):
   * transpose = transpose(input0, input1)
   * transdata = Transdata(transpose)
   * return transdata
   */
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_transdata_split_fraz_nchw", "before");
  std::vector<int64_t> shp{2, 4, 8, 16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto temp = ret->input(1)->cast<CNodePtr>();
  auto transpose = temp->input(1)->cast<CNodePtr>();
  EXPECT_NE(transpose, nullptr);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NCHW"});
  builder.SetInputsDeviceType({kFloat16->type_id()});
  builder.SetOutputsFormat({kOpFormat_C1HWNCoC0});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetInputsReshapeType({""});
  builder.SetOutputsReshapeType({""});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->set_select_kernel_build_info(builder.Build());
  transpose->set_kernel_info(kernel_info);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockInsertTransOpKernelSelectTrans4Dto5D>();
  auto transdata_split_pass = std::make_shared<TransDataSplit>();
  transdata_split_pass->kernel_select_ = std::make_shared<MockTransdataSplitKernelSelect>();
  pm->AddPass(insert_trans_op_pass);
  pm->AddPass(transdata_split_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_transdata_split_fraz_nchw", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWTransdataSplit, test_transdata_split_nchw_fraz) {
  /*
   * def before(input0, input1):
   * transpose = transpose(input0, input1)
   * transdata = Transdata(transpose)
   * return transdata
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_transdata_split_nchw_fraz", "before");
  std::vector<int64_t> shp{2, 4, 8, 16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto temp = ret->input(1)->cast<CNodePtr>();
  auto transpose = temp->input(1)->cast<CNodePtr>();
  EXPECT_NE(transpose, nullptr);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kOpFormat_C1HWNCoC0});
  builder.SetInputsDeviceType({kFloat16->type_id()});
  builder.SetOutputsFormat({"NCHW"});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetInputsReshapeType({""});
  builder.SetOutputsReshapeType({""});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->set_select_kernel_build_info(builder.Build());
  transpose->set_kernel_info(kernel_info);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockInsertTransOpKernelSelectTrans4Dto5D>();
  auto transdata_split_pass = std::make_shared<TransDataSplit>();
  transdata_split_pass->kernel_select_ = std::make_shared<MockTransdataSplitKernelSelect>();
  pm->AddPass(insert_trans_op_pass);
  pm->AddPass(transdata_split_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_transdata_split_nchw_fraz", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
