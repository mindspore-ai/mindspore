/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ir/manager.h"
#include "ir/func_graph.h"

namespace mindspore {
class TestBoostParse : public UT::Common {
 public:
  TestBoostParse() : getPyFun_("gtest_input.pipeline.parse.boost_parse") {}

  virtual void SetUp();

  virtual void TearDown();

  void CheckFalseBranch(const FuncGraphPtr &func_graph, bool folded = true) {
    auto manager = Manage(func_graph);
    EXPECT_TRUE(manager != nullptr);
    bool has_false_branch = false;
    for (auto &fg : manager->func_graphs()) {
      if (fg->debug_info() != nullptr && fg->debug_info()->trace_info() != nullptr) {
        auto symbol = fg->debug_info()->trace_info()->symbol();
        // If contains ✗
        auto contains_false_branch = (symbol.find("\u2717") != std::string::npos);
        if (folded) {
          EXPECT_TRUE(!contains_false_branch);
        } else if (contains_false_branch) {
          has_false_branch = true;
        }
      }
    }
    if (!folded) {
      EXPECT_EQ(has_false_branch, true);
    }
  }

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

void TestBoostParse::SetUp() {}

void TestBoostParse::TearDown() {}

// Feature: Boost parse.
// Description: Parse the network witch has "if var:" statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestIfName) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_if_name", "if_name");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has UnaryOp statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestUnaryOp) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_unary_op", "if_not_name");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestIsNone) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "is_none");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestIsNotNone) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "is_not_none");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestEqualNum) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "equal_num");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestEqualStr) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "equal_str");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should not be folded.
TEST_F(TestBoostParse, TestEqualTensor) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "equal_tensor");
  CheckFalseBranch(func_graph, false);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement with constant condition comment.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestEqualTensorWithComment) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "equal_tensor_with_comment");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestNotEqualNum1) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "not_equal_num1");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestNotEqualNum2) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "not_equal_num2");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestGreater) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "greater");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestGreaterEqual) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "greater_equal");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestLess) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "less");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestLessEqual) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "less_equal");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpNameOrEqual) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "name_or_equal");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpUnaryOpOrEqual) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "unary_op_or_equal");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpNameAndEqual) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "name_and_equal");
  CheckFalseBranch(func_graph);
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation: The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpUnaryOpAndEqual) {
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "unary_op_and_equal");
  CheckFalseBranch(func_graph);
}
}  // namespace mindspore
