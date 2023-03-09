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
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "common/backend_common_test.h"
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/action.h"
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/format_type/check_consistency.h"

namespace mindspore {
namespace opt {
class TestHWCheckConsistency : public BackendCommon {
 public:
  TestHWCheckConsistency() : get_py_fun_("gtest_input.pre_activate.check_consistency", true) {}
  ~TestHWCheckConsistency() override = default;

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWCheckConsistency, test_check_consistency_for_format) {
  // test CheckFormatForConsistency
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_check_consistency", "graph");
  // renormalize func_graph to infer and set shape and type information.
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  g->parameters()[0]->set_abstract(x_abstract);
  auto g_cast = g->get_return()->input(1);
  g_cast->set_abstract(x_abstract);

  // convert to kernel graph
  AbstractBasePtrList args_spec_list;
  auto kernel_graph = GetKernelGraph(g, args_spec_list, false);

  // get make_tuple
  auto ret = kernel_graph->get_return();
  EXPECT_NE(ret, nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_TRUE(ret->input(1)->isa<CNode>());
  auto make_tuple = ret->input(1)->cast<CNodePtr>();

  // set kernel for make tuple
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NCHW"});
  builder1.SetOutputsFormat({"NCHW"});
  builder1.SetInputsDeviceType({kNumberTypeFloat32});
  builder1.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), make_tuple.get());

  // get cast
  EXPECT_NE(make_tuple->input(1), nullptr);
  EXPECT_TRUE(make_tuple->input(1)->isa<CNode>());
  auto cast = make_tuple->input(1)->cast<CNodePtr>();

  // set kernel for cast
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({"NC1HWC0"});
  builder2.SetOutputsFormat({"NCHW"});
  builder2.SetInputsDeviceType({kNumberTypeFloat32});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), cast.get());

  // get para x
  EXPECT_NE(cast->input(1), nullptr);
  EXPECT_TRUE(cast->input(1)->isa<Parameter>());
  auto para = cast->input(1)->cast<ParameterPtr>();

  // set kernel for para x
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder3;
  builder3.SetOutputsFormat({"NCHW"});
  builder3.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(builder3.Build(), para.get());

  // do CheckFormatForConsistency
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::CheckConsistency>());
  optimizer->AddPassManager(pm);
  EXPECT_THROW(optimizer->Optimize(kernel_graph), std::runtime_error);
}
TEST_F(TestHWCheckConsistency, test_check_consistency_for_dtype) {
  // test CheckDataTypeForConsistency
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_check_consistency", "graph");
  // Renormalize func_graph to infer and set shape and type information.
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  g->parameters()[0]->set_abstract(x_abstract);
  auto g_cast = g->get_return()->input(1);
  g_cast->set_abstract(x_abstract);

  // convert to kernel graph
  AbstractBasePtrList args_spec_list;
  auto kernel_graph = GetKernelGraph(g, args_spec_list, false);

  // get make tuple
  auto ret = kernel_graph->get_return();
  EXPECT_NE(ret, nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_TRUE(ret->input(1)->isa<CNode>());
  auto make_tuple = ret->input(1)->cast<CNodePtr>();

  // set kernel for make tuple
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NCHW"});
  builder1.SetOutputsFormat({"NCHW"});
  builder1.SetInputsDeviceType({kNumberTypeFloat32});
  builder1.SetOutputsDeviceType({kNumberTypeFloat16});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), make_tuple.get());

  // get cast
  EXPECT_NE(make_tuple->input(1), nullptr);
  EXPECT_TRUE(make_tuple->input(1)->isa<CNode>());
  auto cast = make_tuple->input(1)->cast<CNodePtr>();

  // set kernel for cast
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({"NCHW"});
  builder2.SetOutputsFormat({"NCHW"});
  builder2.SetInputsDeviceType({kNumberTypeFloat16});
  builder2.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), cast.get());

  // get para x
  EXPECT_NE(cast->input(1), nullptr);
  EXPECT_TRUE(cast->input(1)->isa<Parameter>());
  auto para = cast->input(1)->cast<ParameterPtr>();

  // set kernel for para x
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder3;
  builder3.SetOutputsFormat({"NCHW"});
  builder3.SetOutputsDeviceType({kNumberTypeFloat32});
  AnfAlgo::SetSelectKernelBuildInfo(builder3.Build(), para.get());

  // do CheckFormatForConsistency
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::CheckConsistency>());
  optimizer->AddPassManager(pm);
  EXPECT_THROW(optimizer->Optimize(kernel_graph), std::runtime_error);
}
}  // namespace opt
}  // namespace mindspore
