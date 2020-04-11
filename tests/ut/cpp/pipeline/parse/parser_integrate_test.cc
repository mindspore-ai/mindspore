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
#include "utils/log_adapter.h"
#include "pipeline/parse/parse.h"
#include "debug/draw.h"

namespace mindspore {
namespace parse {

class TestParserIntegrate : public UT::Common {
 public:
  TestParserIntegrate() : getPyFun("gtest_input.pipeline.parse.parser_integrate") {}
  virtual void SetUp();
  virtual void TearDown();
  py::function GetPythonFunction(std::string function);

 public:
  UT::PyFuncGraphFetcher getPyFun;
};

void TestParserIntegrate::SetUp() { UT::InitPythonPath(); }

void TestParserIntegrate::TearDown() {}

TEST_F(TestParserIntegrate, TestParseGraphTestHighOrderFunction) {
  auto func_graph = getPyFun("test_high_order_function");
  ASSERT_TRUE(func_graph != nullptr);
}

TEST_F(TestParserIntegrate, TestParseGraphTestHofTup) {
  auto func_graph = getPyFun("test_hof_tup");
  ASSERT_TRUE(func_graph != nullptr);
}

TEST_F(TestParserIntegrate, TestParseGraphTestWhile2) {
  auto func_graph = getPyFun("test_while_2");
  ASSERT_TRUE(func_graph != nullptr);
}

TEST_F(TestParserIntegrate, TestParseGraphTestNestedClosure) {
  auto func_graph = getPyFun("test_nested_closure");
  ASSERT_TRUE(func_graph != nullptr);
}

TEST_F(TestParserIntegrate, TestParseGraphTestFunctionsInTuples) {
  auto func_graph = getPyFun("test_functions_in_tuples");
  ASSERT_TRUE(func_graph != nullptr);
}

TEST_F(TestParserIntegrate, TestParseGraphTestClosuresInTuples) {
  auto func_graph = getPyFun("test_closures_in_tuples");
  ASSERT_TRUE(func_graph != nullptr);
}

TEST_F(TestParserIntegrate, TestParseGraphTestCompileConv2d) {
  py::function fn_ = python_adapter::GetPyFn("gtest_input.pipeline.parse.parser_integrate", "test_compile_conv2d");
  // fn_();
}

TEST_F(TestParserIntegrate, TestParseGraphTestNone) {
  auto func_graph = getPyFun("test_none");
  ASSERT_TRUE(func_graph != nullptr);
}

TEST_F(TestParserIntegrate, TestParseGraphResolveGetAttr) {
  getPyFun.SetDoResolve(true);
  auto func_graph = getPyFun("test_get_attr");
  draw::Draw("getattr.dot", func_graph);
  ASSERT_TRUE(func_graph != nullptr);
}

/* skip ut test case temporarily
TEST_F(TestParserIntegrate, TestParseGraphResolveUnknown) {
  EXPECT_THROW({ python_adapter::CallPyFn("gtest_input.pipeline.parse.parser_integrate", "test_undefined_symbol"); },
               std::runtime_error);
}
*/

/* #not supported yet
TEST_F(TestParserIntegrate, TestParseGraphTestModelInside) {
    py::function fn_ = python_adapter::GetPyFn(
            "gtest_input.pipeline.parse.parser_integrate", "test_model_inside");
    fn_();

}
 */
/* # infer not supported yet
TEST_F(TestParserIntegrate, TestParseGraphTestTensorAdd) {
    py::function fn_ = python_adapter::GetPyFn(
            "gtest_input.pipeline.parse.parser_integrate", "test_tensor_add");
    fn_();
}

TEST_F(TestParserIntegrate, TestParseGraphTestResnet50Build) {
    py::function fn_ = python_adapter::GetPyFn(
            "gtest_input.pipeline.parse.parser_integrate", "test_resetnet50_build");
    fn_();
}
 */
}  // namespace parse
}  // namespace mindspore
