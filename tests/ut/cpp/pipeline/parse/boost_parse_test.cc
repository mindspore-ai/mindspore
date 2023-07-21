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
#include "utils/ms_utils.h"

namespace mindspore {
class TestBoostParse : public UT::Common {
 public:
  TestBoostParse() : getPyFun_("gtest_input.pipeline.parse.boost_parse") {}

  virtual void SetUp();

  virtual void TearDown();

  void CheckHasFalseBranch(const FuncGraphPtr &func_graph) {
    auto manager = Manage(func_graph);
    EXPECT_TRUE(manager != nullptr);
    for (auto &fg : manager->func_graphs()) {
      if (fg->debug_info() != nullptr && fg->debug_info()->trace_info() != nullptr) {
        auto symbol = fg->debug_info()->trace_info()->symbol();
        // ✓ or ↓
        EXPECT_TRUE(symbol == "\u2713" || symbol == "\u2193");
      }
    }
  }

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

void TestBoostParse::SetUp() {}

void TestBoostParse::TearDown() {}

// Feature: Boost parse.
// Description: Parse the network witch has "if var:" statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestIfName) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_if_name", "if_name");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has UnaryOp statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestUnaryOp) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_unary_op", "if_not_name");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestIsNone) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "is_none");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestIsNotNone) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "is_not_none");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestNotEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "not_equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestGreater) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "greater");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestGreaterEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "greater_equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestLess) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "less");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has comparison statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestLessEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_compare", "less_equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpNameOrEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "name_or_equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpUnaryOpOrEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "unary_op_or_equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpNameAndEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "name_and_equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}

// Feature: Boost parse.
// Description: Parse the network witch has BoolOp statement.
// Expectation:The false branch should be folded.
TEST_F(TestBoostParse, TestBoolOpUnaryOpAndEqual) {
  common::SetEnv("MS_DEV_BOOST_PARSE", "1");
  FuncGraphPtr func_graph = getPyFun_.CallAndParseRet("test_bool_op", "unary_op_and_equal");
  CheckHasFalseBranch(func_graph);
  common::SetEnv("MS_DEV_BOOST_PARSE", "0");
}
}  // namespace mindspore
