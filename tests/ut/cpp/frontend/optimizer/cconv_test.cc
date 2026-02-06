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
#include "common/py_func_graph_fetcher.h"

#include "ir/func_graph_cloner.h"
#include "utils/log_adapter.h"
#include "frontend/jit/ps/parse/parse.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"

namespace mindspore {
void CheckNoFreeVariables(FuncGraphPtr root) {
  auto mng = Manage(root);
  for (auto &iter : mng->func_graphs()) {
    auto g = iter;
    if (g == nullptr) {
      continue;
    }
    ASSERT_TRUE(g->parent() == nullptr);

    auto nodes = g->nodes();
    for (auto &node : nodes) {
      ASSERT_EQ(node->func_graph(), g);
      auto cnode = node->cast<CNodePtr>();
      if (cnode != nullptr) {
        for (auto &weak_input : cnode->weak_inputs()) {
          auto inp = weak_input.lock();
          MS_EXCEPTION_IF_NULL(inp);
          ASSERT_TRUE(inp->func_graph() == nullptr || inp->func_graph() == g);
        }
      }
    }
  }
}

void CheckCconv(FuncGraphPtr g) {
  auto mng = Manage(g);
  auto new_g = LiftingClone(g);
  CheckNoFreeVariables(new_g);
}

class TestCconv : public UT::Common {
 public:
  TestCconv() : getPyFun("gtest_input.optimizer.cconv_test") {}

  virtual void SetUp();

 public:
  UT::PyFuncGraphFetcher getPyFun;
};

void TestCconv::SetUp() {}

TEST_F(TestCconv, TestStraight) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_straight");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestSimpleClosure) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_simple_closure");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestMax) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_max");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestDeepNesting) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_deep_nesting");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestReturnInDoubleWhile) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_return_in_double_while");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestPow10) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_pow10");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestClosureAsSimpleFv) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_closure_as_simple_fv");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestClosureAsFv) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_closure_as_fv");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestClosureAsDoubleFv) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_closure_as_double_fv");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestClosureLiftSameParam) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_closure_lift_same_param");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestClosureAsLoop) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_closure_as_loop");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

TEST_F(TestCconv, TestClosureLiftCNode) {
  FuncGraphPtr func_graph = getPyFun.CallAndParseRet("get_test_cconv_fn", "test_closure_lift_cnode");
  ASSERT_TRUE(nullptr != func_graph);
  CheckCconv(func_graph);
}

}  // namespace mindspore
