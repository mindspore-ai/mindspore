/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "include/backend/anf_runtime_algorithm.h"
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "include/backend/optimizer/optimizer.h"
#include "ir/param_info.h"
#include "include/common/utils/anfalgo.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/enhancer/insert_tensor_move_for_hccl_op.h"
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#undef private
#undef protected
namespace mindspore {
namespace opt {
class TestHWInsertTensorMoveForHccl : public BackendCommon {
 public:
  TestHWInsertTensorMoveForHccl() : get_py_fun_("gtest_input.pre_activate.insert_tensor_move_for_hccl_op", true) {}
  ~TestHWInsertTensorMoveForHccl() override = default;

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

class MockInsertTensorMoveForHcclKernelQuery : public KernelQuery {
 public:
  MockInsertTensorMoveForHcclKernelQuery() = default;
  ~MockInsertTensorMoveForHcclKernelQuery() override = default;
  bool IsTbeRef(const AnfNodePtr &node) override {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      return false;
    }
    auto node_name = common::AnfAlgo::GetCNodeName(node->cast<CNodePtr>());
    return node_name == "ApplyMomentum" || node_name == "AssignAdd";
  }
};

TEST_F(TestHWInsertTensorMoveForHccl, test_cond1_no_insert) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond1", "before2");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);
  auto origin_graph = std::make_shared<session::KernelGraph>(*kg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertTensorMoveForHcclOp>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  EXPECT_TRUE(CheckEqualGraph(origin_graph, new_graph));
}

TEST_F(TestHWInsertTensorMoveForHccl, test_cond2) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond2", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);
  for (auto p : kg->parameters()) {
    auto param = p->cast<ParameterPtr>();
    EXPECT_NE(param, nullptr);
  }

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertTensorMoveForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertTensorMoveForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond2", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWInsertTensorMoveForHccl, test_cond3) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond3", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{3, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertTensorMoveForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertTensorMoveForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond3", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWInsertTensorMoveForHccl, test_cond4) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond4", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  for (auto p : kg->parameters()) {
    auto param = p->cast<ParameterPtr>();
    EXPECT_NE(param, nullptr);
  }

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::InsertTensorMoveForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertTensorMoveForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond4", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWInsertTensorMoveForHccl, test_cond5) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond5", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);

  for (auto p : kg->parameters()) {
    auto param = p->cast<ParameterPtr>();
    EXPECT_NE(param, nullptr);
  }

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  // This pass run before hccl_pass to unfold inputs of hccl node
  pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>());
  auto pass = std::make_shared<opt::InsertTensorMoveForHcclOp>();
  pass->kernel_query_ = std::make_shared<MockInsertTensorMoveForHcclKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(kg);
  kg->SetExecOrderByDefault();

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_insert_tensor_move_for_hccl_op_cond5", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
