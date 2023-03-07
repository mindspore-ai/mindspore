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
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "pipeline/jit/resource.h"
#include "frontend/operator/ops.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "backend/common/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_getnext.h"

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestHWInsertTensorMoveForGetNext : public BackendCommon {
 public:
  TestHWInsertTensorMoveForGetNext() : get_py_fun_("gtest_input.pre_activate.insert_tensor_move_for_getnext", true) {}
  ~TestHWInsertTensorMoveForGetNext() override = default;

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWInsertTensorMoveForGetNext, test_insert_tensor_move_for_getnext_multi_output) {
  FuncGraphPtr g_before = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_getnext", "getnext_multi_output_before");

  AbstractBasePtrList args_spec_list{};
  auto kernel_graph = GetKernelGraph(g_before, args_spec_list);

  KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({kFloat32->type_id(), kInt32->type_id()});
  auto ret = kernel_graph->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>()->input(1), nullptr);
  auto get_next = ret->input(1)->cast<CNodePtr>()->input(1);
  get_next->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), get_next.get());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::InsertTensorMoveForGetNext>());
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kernel_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_getnext", "getnext_multi_output_after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore