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
#include "include/backend/anf_runtime_algorithm.h"
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "include/backend/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/enhancer/getnext_tensor_move_elimination.h"

namespace mindspore {
namespace opt {
class TestGetNextTensorMoveElimination : public BackendCommon {
 public:
  TestGetNextTensorMoveElimination() : get_py_fun_("gtest_input.pre_activate.getnext_tensor_move_elimination_test", true) {}

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestGetNextTensorMoveElimination, test_getnext_tensormove_elimination) {
  FuncGraphPtr g_before = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination", "before");
  ASSERT_TRUE(g_before != nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::GetnextTensorMoveElimination>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(g_before);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestGetNextTensorMoveElimination, test_getnext_tensor_move_elimination_no_attr) {
  FuncGraphPtr g_before = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination_no_attr", "before");
  ASSERT_TRUE(g_before != nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::GetnextTensorMoveElimination>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(g_before);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination_no_attr", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestGetNextTensorMoveElimination, test_getnext_tensor_move_elimination_tensor_move_multi_users) {
  FuncGraphPtr g_before = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination_tensor_move_multi_users", "before");
  ASSERT_TRUE(g_before != nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::GetnextTensorMoveElimination>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(g_before);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination_tensor_move_multi_users", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestGetNextTensorMoveElimination, test_getnext_tensor_move_elimination_next_multi_inputs) {
  FuncGraphPtr g_before = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination_next_multi_inputs", "before");
  ASSERT_TRUE(g_before != nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::GetnextTensorMoveElimination>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(g_before);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_getnext_tensor_move_elimination_next_multi_inputs", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

}  // namespace opt
}  // namespace mindspore
