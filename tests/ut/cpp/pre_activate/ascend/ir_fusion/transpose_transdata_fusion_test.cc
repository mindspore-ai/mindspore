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
#include "device/kernel_info.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/oplib/oplib.h"
#define private public
#define protected public
#include "pre_activate/ascend/format_type/insert_trans_op.h"
#include "pre_activate/ascend/ir_fusion/transpose_transdata_fusion.h"
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

class MockInsertTransOpKernelSelectTrans4Dto5D : public KernelSelect {
 public:
  MockInsertTransOpKernelSelectTrans4Dto5D() = default;
  ~MockInsertTransOpKernelSelectTrans4Dto5D() override = default;
  void SelectKernel(const CNodePtr &cnode) override {
    if (AnfAlgo::GetCNodeName(cnode) == "TransData") {
      KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({"NCHW"});
      builder.SetInputsDeviceType({kFloat16->type_id()});
      builder.SetOutputsFormat({"NC1HWC0"});
      builder.SetOutputsDeviceType({kFloat16->type_id()});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
    } else {
      KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({"NC1HWC0"});
      builder.SetInputsDeviceType({kFloat16->type_id()});
      builder.SetOutputsFormat({"NC1HWC0"});
      builder.SetOutputsDeviceType({kFloat16->type_id()});
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
    }
  }
};

class MockTransposeTransdataFusionKernelSelect : public KernelSelect {
 public:
  MockTransposeTransdataFusionKernelSelect() = default;
  ~MockTransposeTransdataFusionKernelSelect() override = default;
  bool CheckKernelAccuracySupported(const CNodePtr &kernel_node,
                                    const kernel::KernelBuildInfoPtr &new_kernel_build_info) override {
    std::vector<std::shared_ptr<kernel::KernelBuildInfo>> kernel_info_list;
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetInputsFormat({kOpFormat_NCHW});
    builder.SetOutputsFormat({kOpFormat_DEFAULT});
    builder.SetInputsDeviceType({kNumberTypeFloat16});
    builder.SetOutputsDeviceType({kNumberTypeFloat16});
    builder.SetKernelType(KernelType::AUTO_DIFF_KERNEL);
    builder.SetFusionType(kernel::FusionType::OPAQUE);
    builder.SetProcessor(kernel::Processor::AICORE);
    kernel_info_list.push_back(builder.Build());
    MS_LOG(INFO) << "transpose transdata fusion success";
    MS_LOG(INFO) << "new transdata build info input format:" << new_kernel_build_info->GetInputFormat(0)
                 << ",outputformat:" << new_kernel_build_info->GetOutputFormat(0)
                 << ",kerneltype:" << new_kernel_build_info->kernel_type()
                 << ",fusiontype:" << new_kernel_build_info->fusion_type()
                 << ",process:" << new_kernel_build_info->processor();
    auto result = std::find_if(kernel_info_list.begin(), kernel_info_list.end(),
                               [&new_kernel_build_info](kernel::KernelBuildInfoPtr item) {
                                 MS_EXCEPTION_IF_NULL(item);
                                 return *item == *new_kernel_build_info;
                               });
    return result != kernel_info_list.end();
  }
};

TEST_F(TestHWTransposeTransdataFusion, test_transpose_transdata_fusion) {
  /*
   * def before(input0, input1):
   * transpose = transpose(input0, input1)
   * transdata = Transdata(transpose)
   * return transdata
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_transpose_transdata_fusion", "before");
  std::vector<int> shp{2, 4, 8, 16};
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
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetOutputsDeviceType({kFloat16->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  auto kernel_info = std::make_shared<device::KernelInfo>();
  kernel_info->set_select_kernel_build_info(builder.Build());
  transpose->set_kernel_info(kernel_info);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto insert_trans_op_pass = std::make_shared<opt::InsertTransOp>();
  insert_trans_op_pass->kernel_select_ = std::make_shared<MockInsertTransOpKernelSelectTrans4Dto5D>();
  pm->AddPass(insert_trans_op_pass);
  auto transpose_transdata_pass = std::make_shared<opt::TransposeTransDataFusion>();
  transpose_transdata_pass->kernel_select_ = std::make_shared<MockTransposeTransdataFusionKernelSelect>();
  pm->AddPass(transpose_transdata_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

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
