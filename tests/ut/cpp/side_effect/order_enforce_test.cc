/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <memory>

#include "pipeline/jit/ps/static_analysis/order_enforce.h"

#include "mindspore/core/ops/framework_ops.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "ir/visitor.h"
#include "pipeline/jit/ps/action.h"
#include "ir/func_graph.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestSideEffectOrderEnforce : public UT::Common {
 public:
  TestSideEffectOrderEnforce() : getPyFun("gtest_input.side_effect.order_enforce_test", true) {}

  void SetUp() {}

 public:
  UT::PyFuncGraphFetcher getPyFun;
};

void InsertUNode(const FuncGraphPtr &func_graph) {
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeper);
  for (const auto &node : nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
      auto load = node->cast<CNodePtr>();
      load->add_input(NewValueNode(kUMonad));
    }
  }
}

size_t CountTensorMoveNum(const FuncGraphPtr &func_graph) {
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  auto accumulate_func = [](size_t prev_num, const AnfNodePtr &node) -> size_t {
    return IsPrimitiveCNode(node, prim::kPrimTensorMove) ? prev_num + 1 : prev_num;
  };
  return std::accumulate(nodes.begin(), nodes.end(), 0, accumulate_func);
}

std::vector<size_t> GetInsertTensorMoveIndexes(const FuncGraphPtr &func_graph) {
  HashMap<AnfNodePtr, size_t> load_index;
  std::vector<size_t> indexes;
  size_t cur_index = 0;
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (const auto &node : nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
      load_index[node] = cur_index++;
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimTensorMove)) {
      auto tensor_move = node->cast<CNodePtr>();
      auto load = tensor_move->input(1);
      auto it = load_index.find(load);
      if (it == load_index.end()) {
        MS_LOG(EXCEPTION) << "Can't find load as input of tensor move.";
      }
      indexes.push_back(it->second);
    }
  }
  return indexes;
}

// Feature: Insert tensor move.
// Description: If there are 2 or more `Load` nodes with same ref key, `TensorMove` nodes need inserted after these
// `Load` nodes.
// Expectation: `TensorMove` nodes inserted after `Load` nodes with same ref key of which nums greater than 2.
TEST_F(TestSideEffectOrderEnforce, TwoLoads) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_order_enforce", "test_two_loads");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(test_graph);
  std::vector<AbstractBasePtr> args_spec;

  // Execute pass
  InsertUNode(test_graph);
  pipeline::AbstractSpecializeAction(res);
  pipeline::OrderEnforce(res->func_graph());

  // Check tensor move indexes.
  auto tensor_move_indexes = GetInsertTensorMoveIndexes(res->func_graph());
  std::vector<size_t> expect_indexes = {0, 1, 2};
  ASSERT_EQ(tensor_move_indexes.size(), expect_indexes.size());
  auto result = tensor_move_indexes == expect_indexes;
  ASSERT_TRUE(result);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Insert tensor move.
// Description: If the arg of `Partial` or func graph call node is `Load`, insert a `TensorMove` node after it.
// Expectation: `TensorMove` nodes inserted after `Load` nodes of which are arg of `Partial` or func graph call node.
TEST_F(TestSideEffectOrderEnforce, PartialArgCallArg) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_order_enforce", "test_partial_load_arg");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(test_graph);
  std::vector<AbstractBasePtr> args_spec;

  // Execute pass
  InsertUNode(test_graph);
  pipeline::AbstractSpecializeAction(res);
  pipeline::OrderEnforce(res->func_graph());

  // Check tensor move indexes.
  auto tensor_move_indexes = GetInsertTensorMoveIndexes(res->func_graph());
  std::vector<size_t> expect_indexes = {0, 1};
  ASSERT_EQ(tensor_move_indexes.size(), expect_indexes.size());
  auto result = tensor_move_indexes == expect_indexes;
  ASSERT_TRUE(result);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Insert tensor move.
// Description: If the arg of `Partial` or func graph call node is `Load`, insert a `TensorMove` node after it.In this
// case, an arg is output of func graoh call.
// Expectation: `TensorMove` nodes inserted after `Load` nodes of which are arg of `Partial` or func graph call node.
TEST_F(TestSideEffectOrderEnforce, CallOutAsArg) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_order_enforce", "test_partial_load_arg_call_out_as_arg");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(test_graph);
  std::vector<AbstractBasePtr> args_spec;

  // Execute pass
  InsertUNode(test_graph);
  pipeline::AbstractSpecializeAction(res);
  pipeline::OrderEnforce(res->func_graph());

  // Check tensor move indexes.
  auto tensor_move_indexes = GetInsertTensorMoveIndexes(res->func_graph());
  std::vector<size_t> expect_indexes = {0, 1};
  ASSERT_EQ(tensor_move_indexes.size(), expect_indexes.size());
  auto result = tensor_move_indexes == expect_indexes;
  ASSERT_TRUE(result);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Insert tensor move.
// Description: A func graph call node output is abstract ref and as a input of load.
// Expectation: `TensorMove` nodes inserted after these `Load` nodes.
TEST_F(TestSideEffectOrderEnforce, DISABLED_CallOutLoad) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_order_enforce", "test_call_out_load");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(test_graph);
  std::vector<AbstractBasePtr> args_spec;

  // Execute pass
  InsertUNode(test_graph);
  pipeline::AbstractSpecializeAction(res);
  pipeline::OrderEnforce(res->func_graph());

  // Check tensor move indexes.
  auto tensor_move_indexes = GetInsertTensorMoveIndexes(res->func_graph());
  std::vector<size_t> expect_indexes = {0, 1};
  ASSERT_EQ(tensor_move_indexes.size(), expect_indexes.size());
  auto result = tensor_move_indexes == expect_indexes;
  ASSERT_TRUE(result);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Insert tensor move.
// Description: A func graph call node output is abstract ref and as a input of load and has a same refkey with another
// simple `Load` node.
// Expectation: `TensorMove` nodes inserted after these `Load` nodes whose input is a func graph call node and after the
// same refkey simple loads.
TEST_F(TestSideEffectOrderEnforce, DISABLED_LoadRefSameToSwitchCallOutGetItemGetItem) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_order_enforce", "load_ref_same_to_call_out");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(test_graph);
  std::vector<AbstractBasePtr> args_spec;

  // Execute pass
  InsertUNode(test_graph);
  pipeline::AbstractSpecializeAction(res);
  pipeline::OrderEnforce(res->func_graph());

  // Check tensor move indexes.
  auto tensor_move_indexes = GetInsertTensorMoveIndexes(res->func_graph());
  std::vector<size_t> expect_indexes = {0, 1, 3, 4};
  ASSERT_EQ(tensor_move_indexes.size(), expect_indexes.size());
  auto result = tensor_move_indexes == expect_indexes;
  ASSERT_TRUE(result);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Insert tensor move.
// Description: A func graph call node output is abstract ref and as a input of load and has a same refkey with another
// simple `Load` node.In this case, call node is a switch_switch call node.
// Expectation: `TensorMove` nodes inserted after these `Load` nodes whose input is a func graph call node and after the
// same refkey simple loads.
TEST_F(TestSideEffectOrderEnforce, DISABLED_LoadCallSwitchCallSwitchCallOut) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_order_enforce", "test_switch_switch_call");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(test_graph);
  std::vector<AbstractBasePtr> args_spec;

  // Execute pass
  InsertUNode(test_graph);
  pipeline::AbstractSpecializeAction(res);
  pipeline::OrderEnforce(res->func_graph());
  // Check tensor move indexes.
  auto tensor_move_indexes = GetInsertTensorMoveIndexes(res->func_graph());
  std::vector<size_t> expect_indexes = {0, 1, 2, 4};
  ASSERT_EQ(tensor_move_indexes.size(), expect_indexes.size());
  auto result = tensor_move_indexes == expect_indexes;
  ASSERT_TRUE(result);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}
}  // namespace opt
}  // namespace mindspore
