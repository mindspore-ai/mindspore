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
#include <iostream>
#include <string>
#include "common/common_test.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "common/py_func_graph_fetcher.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "include/common/debug/draw.h"

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/ps/action.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace parse {
class TestParallelIf : public UT::Common {
 public:
  TestParallelIf() : getPyFun("gtest_input.pipeline.parse.parallel_if") {}
  virtual void SetUp();
  virtual void TearDown();
  py::function GetPythonFunction(std::string function);

  bool CheckIsomorphic(FuncGraphPtr basic, FuncGraphPtr manual, std::vector<opt::SubstitutionPtr> opts = {}) {
    opt::SubstitutionList transform(opts);
    FuncGraphPairMapEquiv equiv_graph;
    NodeMapEquiv equiv_node;

    opt::OptimizerPtr optimizer = std::make_shared<opt::Optimizer>("ut_test", std::make_shared<pipeline::Resource>());
    FuncGraphPtr basic_clone = BasicClone(basic);
    transform(basic_clone, optimizer);
    FuncGraphPtr manual_clone = BasicClone(manual);
    transform(manual_clone, optimizer);

    return Isomorphic(basic_clone, manual_clone, &equiv_graph, &equiv_node);
  }

  void CheckParallelIfTransform(const std::string &test_case) {
    FuncGraphPtr basic_graph = getPyFun.CallAndParseRet(test_case, "basic");
    ASSERT_TRUE(basic_graph != nullptr);
    FuncGraphPtr manual_graph = getPyFun.CallAndParseRet(test_case, "manual");
    ASSERT_TRUE(manual_graph != nullptr);

    pipeline::ResourcePtr res1 = std::make_shared<pipeline::Resource>();

    tensor::TensorPtr x_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{1});
    tensor::TensorPtr y_tensor = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{1});

    AbstractBasePtr abstract_x = abstract::FromValue(x_tensor, true);
    AbstractBasePtr abstract_y = abstract::FromValue(y_tensor, true);
    abstract::AbstractBasePtrList args_spec_list{abstract_x, abstract_y};

    abstract::AnalysisResult result = pipeline::AbstractAnalyze(res1, basic_graph, args_spec_list);
    auto new_basic_graph = pipeline::ProgramSpecialize(res1, basic_graph, result.context);

    pipeline::ResourcePtr res2 = std::make_shared<pipeline::Resource>();
    result = pipeline::AbstractAnalyze(res2, manual_graph, args_spec_list);
    auto new_manual_graph = pipeline::ProgramSpecialize(res2, manual_graph, result.context);

    auto patterns = std::vector<opt::SubstitutionPtr>({irpass_lib_.inline_, irpass_lib_.switch_simplify_});
    ASSERT_TRUE(CheckIsomorphic(new_basic_graph, new_manual_graph, patterns));

    abstract::AnalysisResultCacheMgr::GetInstance().Clear();
    abstract::AnalysisContext::ClearContext();
  }

  void CheckParallelIfTransformationCount(const std::string &test_case, int expected_count) {
    FuncGraphPtr func_graph = getPyFun.CallAndParseRet(test_case);
    ASSERT_TRUE(func_graph != nullptr);

    int count = 0;

    auto manager = mindspore::Manage(func_graph, true);

    // Get user cnode of all switch cnode: switch(cond, branch1, branch2)();
    AnfNodePtrList switch_cnodes_user;
    const auto &node_users = manager->node_users();
    for (const auto &node : manager->all_nodes()) {
      if (IsPrimitiveCNode(node, prim::kPrimSwitch)) {
        auto switch_cnode_user_iter = node_users.find(node);
        if (switch_cnode_user_iter != node_users.end()) {
          ASSERT_EQ(switch_cnode_user_iter->second.size(), 1);
          auto switch_cnode_user = switch_cnode_user_iter->second.front().first;
          switch_cnodes_user.push_back(switch_cnode_user);
        }
      }
    }
    // Check if the switch_cnode_user is used by GetItem call or FuncGraph call.
    for (const auto &switch_cnode_user : switch_cnodes_user) {
      auto user_iter = node_users.find(switch_cnode_user);
      if (user_iter != node_users.end()) {
        ASSERT_GE(user_iter->second.size(), 1);
        auto user = user_iter->second.front().first;
        if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem)) {
          count++;
        } else if (GetCNodeFuncGraph(user) != nullptr) {
          count++;
        }
      }
    }
    ASSERT_EQ(count, expected_count);
  }

 public:
  UT::PyFuncGraphFetcher getPyFun;
  opt::irpass::OptimizeIRPassLib irpass_lib_;
};

void TestParallelIf::SetUp() { UT::InitPythonPath(); }

void TestParallelIf::TearDown() {}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with single if/else.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, SimpleIf) { CheckParallelIfTransform("test_simple_if"); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with if-by-if.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, IfByIf) { CheckParallelIfTransform("test_if_by_if"); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with if-in-if.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, IfInIf) { CheckParallelIfTransform("test_if_in_if"); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for test code with if-elif-else.
// Expectation: The funcgraph after transformation should be isomorphic with the funcgraph manually constructed.
TEST_F(TestParallelIf, IfElifElse) { CheckParallelIfTransform("test_if_elif_else"); }

// Return statement section.
// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while return).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileReturnInElse) { CheckParallelIfTransformationCount("test_while_return_in_else", 0); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_return_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if return/else break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnElseBreakInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_return_else_break_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if return/else return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnElseReturnInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_return_else_return_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileReturnInWhileInElse) {
  CheckParallelIfTransformationCount("test_while_return_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(if return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnInWhileInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_return_in_while_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(if return/else return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnElseReturnInWhileInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_return_else_return_in_while_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if/else by while(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileReturnAfterIfElseInElse) {
  CheckParallelIfTransformationCount("test_while_return_after_if_else_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(return) by if/else).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterWhileReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_while_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if/else by if(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnAfterIfElseInElse) {
  CheckParallelIfTransformationCount("test_if_return_after_if_else_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if(return) by if/else).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterIfReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_if_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else by if/else(while(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileReturnInElseAfterIfElse) {
  CheckParallelIfTransformationCount("test_while_return_in_else_after_if_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(return)) by if/else.
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterByWhileReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_by_while_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else by if/else(if(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnInElseAfterIfElse) {
  CheckParallelIfTransformationCount("test_if_return_in_else_after_if_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if(return)) by if/else.
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterByIfReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_by_if_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if/else)/else(while(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseInIfWhileReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_in_if_while_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if/else)/else(if(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseInIfIfReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_in_if_if_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for return).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForReturnInElse) { CheckParallelIfTransformationCount("test_for_return_in_else", 0); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnInForInElse) {
  CheckParallelIfTransformationCount("test_if_return_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if return/else break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnElseBreakInForInElse) {
  CheckParallelIfTransformationCount("test_if_return_else_break_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if return/else return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnElseReturnInForInElse) {
  CheckParallelIfTransformationCount("test_if_return_else_return_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForReturnInForInElse) {
  CheckParallelIfTransformationCount("test_for_return_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(if return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnInForInForInElse) {
  CheckParallelIfTransformationCount("test_if_return_in_for_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(if return/else return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnElseReturnInForInForInElse) {
  CheckParallelIfTransformationCount("test_if_return_else_return_in_for_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if/else by for(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForReturnAfterIfElseInElse) {
  CheckParallelIfTransformationCount("test_for_return_after_if_else_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(return) by if/else).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterForReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_for_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else by if/else(for(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForReturnInElseAfterIfElse) {
  CheckParallelIfTransformationCount("test_for_return_in_else_after_if_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(return)) by if/else.
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterByForReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_by_for_return_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if/else)/else(for(return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseInIfForReturnInElse) {
  CheckParallelIfTransformationCount("test_if_else_in_if_for_return_in_else", 1);
}

// Break statement section.
// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while break).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileBreakInElse) { CheckParallelIfTransformationCount("test_while_break_in_else", 1); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_break_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if break/else break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakElseBreakInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_break_else_break_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if break/else return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakElseReturnInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_break_else_return_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(break))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileBreakInWhileInElse) {
  CheckParallelIfTransformationCount("test_while_break_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(if break))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakInWhileInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_break_in_while_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(if break/else return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakElseReturnInWhileInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_break_else_return_in_while_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if/else by while(break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileBreakAfterIfElseInElse) {
  CheckParallelIfTransformationCount("test_while_break_after_if_else_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(break) by if/else).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterWhileBreakInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_while_break_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else by if/else(while(break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileBreakInElseAfterIfElse) {
  CheckParallelIfTransformationCount("test_while_break_in_else_after_if_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(break)) by if/else.
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterByWhileBreakInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_by_while_break_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if/else)/else(while(break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseInIfWhileBreakInElse) {
  CheckParallelIfTransformationCount("test_if_else_in_if_while_break_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for break).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForBreakInElse) { CheckParallelIfTransformationCount("test_for_break_in_else", 1); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakInForInElse) {
  CheckParallelIfTransformationCount("test_if_break_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if break/else return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakElseReturnInForInElse) {
  CheckParallelIfTransformationCount("test_if_break_else_return_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if break/else break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakElseBreakInForInElse) {
  CheckParallelIfTransformationCount("test_if_break_else_break_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(break))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForBreakInForInElse) {
  CheckParallelIfTransformationCount("test_for_break_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(if break))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakInForInForInElse) {
  CheckParallelIfTransformationCount("test_if_break_in_for_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(if break/else return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfBreakElseReturnInForInForInElse) {
  CheckParallelIfTransformationCount("test_if_break_else_return_in_for_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if/else by for(break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForBreakAfterIfElseInElse) {
  CheckParallelIfTransformationCount("test_for_break_after_if_else_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(break) by if/else).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterForBreakInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_for_break_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else by if/else(for(break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForBreakInElseAfterIfElse) {
  CheckParallelIfTransformationCount("test_for_break_in_else_after_if_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(break)) by if/else.
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterByForBreakInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_by_for_break_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if/else)/else(for(break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseInIfForBreakInElse) {
  CheckParallelIfTransformationCount("test_if_else_in_if_for_break_in_else", 2);
}

// Continue statement section
// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while continue).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileContinueInElse) {
  CheckParallelIfTransformationCount("test_while_continue_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_continue_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if continue/else continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueElseContinueInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_continue_else_continue_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(if continue/else return)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueElseReturnInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_continue_else_return_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(continue))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileContinueInWhileInElse) {
  CheckParallelIfTransformationCount("test_while_continue_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(if continue))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueInWhileInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_continue_in_while_in_while_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(while(if continue/else return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueElseReturnInWhileInWhileInElse) {
  CheckParallelIfTransformationCount("test_if_continue_else_return_in_while_in_while_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if/else by while(continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileContinueAfterIfElseInElse) {
  CheckParallelIfTransformationCount("test_while_continue_after_if_else_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(continue) by if/else).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterWhileContinueInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_while_continue_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else by if/else(while(continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountWhileContinueInElseAfterIfElse) {
  CheckParallelIfTransformationCount("test_while_continue_in_else_after_if_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(while(continue)) by if/else.
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterByWhileContinueInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_by_while_continue_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if/else)/else(while(continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseInIfWhileContinueInElse) {
  CheckParallelIfTransformationCount("test_if_else_in_if_while_continue_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for continue).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForContinueInElse) { CheckParallelIfTransformationCount("test_for_continue_in_else", 1); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueInForInElse) {
  CheckParallelIfTransformationCount("test_if_continue_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if return/else continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfReturnElseContinueInForInElse) {
  CheckParallelIfTransformationCount("test_if_return_else_continue_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(if continue/else continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueElseContinueInForInElse) {
  CheckParallelIfTransformationCount("test_if_continue_else_continue_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(continue))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForContinueInForInElse) {
  CheckParallelIfTransformationCount("test_for_continue_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(if continue))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueInForInForInElse) {
  CheckParallelIfTransformationCount("test_if_continue_in_for_in_for_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(for(if continue/else return))).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfContinueElseReturnInForInForInElse) {
  CheckParallelIfTransformationCount("test_if_continue_else_return_in_for_in_for_in_else", 0);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(if/else by for(continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForContinueAfterIfElseInElse) {
  CheckParallelIfTransformationCount("test_for_continue_after_if_else_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(continue) by if/else).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterForContinueInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_for_continue_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else by if/else(for(continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountForContinueInElseAfterIfElse) {
  CheckParallelIfTransformationCount("test_for_continue_in_else_after_if_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if/else(for(continue)) by if/else.
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseAfterByForContinueInElse) {
  CheckParallelIfTransformationCount("test_if_else_after_by_for_continue_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if/else)/else(for(continue)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountIfElseInIfForContinueInElse) {
  CheckParallelIfTransformationCount("test_if_else_in_if_for_continue_in_else", 2);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(func call)/else(while(break)).
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountFuncCallInIfWhileBreakInElse) {
  CheckParallelIfTransformationCount("test_func_call_in_if_while_break_in_else", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for while(if(if/if(break)))
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountFuncCallIfByIfBreakInIfInWhile) {
  CheckParallelIfTransformationCount("test_if_by_if_break_in_if_in_while", 1);
}

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if(raise))/else
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountFuncCallIfRaiseRaise) { CheckParallelIfTransformationCount("test_if_raise_raise", 1); }

// Feature: Parallel if transformation
// Description: Check parallel if transformatin for if(if(assert)/else)
// Expectation: The count of parallel if transformation should be equal to the expected count.
TEST_F(TestParallelIf, CountFuncCallIfAssertFailure) {
  CheckParallelIfTransformationCount("test_if_assert_failure", 2);
}
}  // namespace parse
}  // namespace mindspore
