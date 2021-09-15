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
#include "pipeline/jit/parse/parse.h"
#include "debug/draw.h"

namespace mindspore {
namespace parse {

class TestParserPrimitive : public UT::Common {
 public:
  TestParserPrimitive() {}
  virtual void SetUp();
  virtual void TearDown();
};

void TestParserPrimitive::SetUp() { UT::InitPythonPath(); }

void TestParserPrimitive::TearDown() {}

TEST_F(TestParserPrimitive, TestParserOpsMethod1) {
  py::function fn_ = python_adapter::GetPyFn("gtest_input.pipeline.parse.parse_primitive", "test_ops_f1");

  FuncGraphPtr func_graph = ParsePythonCode(fn_);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);
}

TEST_F(TestParserPrimitive, TestParserOpsMethod2) {
  py::function fn_ = python_adapter::GetPyFn("gtest_input.pipeline.parse.parse_primitive", "test_ops_f2");

  FuncGraphPtr func_graph = ParsePythonCode(fn_);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);
}

// Test primitive class obj
TEST_F(TestParserPrimitive, TestParsePrimitive) {
#if 0  // Segmentation fault
  py::object obj_ = python_adapter::CallPyFn("gtest_input.pipeline.parse.parse_primitive", "test_primitive_obj");
  Parser::InitParserEnvironment(obj_);
  FuncGraphPtr func_graph = ParsePythonCode(obj_);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);
#endif
}

/* skip ut test case temporarily
TEST_F(TestParserPrimitive, TestParsePrimitiveParmeter) {
  py::object obj_ =
    python_adapter::CallPyFn("gtest_input.pipeline.parse.parse_primitive", "test_primitive_obj_parameter");
  Parser::InitParserEnvironment(obj_);
  FuncGraphPtr func_graph = ParsePythonCode(obj_);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);
}

TEST_F(TestParserPrimitive, TestParsePrimitiveParmeter2) {
  py::object obj_ = python_adapter::CallPyFn("gtest_input.pipeline.parse.parse_primitive", "test_primitive_functional");
  Parser::InitParserEnvironment(obj_);
  FuncGraphPtr func_graph = ParsePythonCode(obj_);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);
}
*/

}  // namespace parse
}  // namespace mindspore
