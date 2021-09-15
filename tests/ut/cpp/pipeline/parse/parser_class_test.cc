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

class TestParserClass : public UT::Common {
 public:
  TestParserClass() {}
  virtual void SetUp();
  virtual void TearDown();
};

void TestParserClass::SetUp() { UT::InitPythonPath(); }

void TestParserClass::TearDown() {}

// Test case1 : test class method
TEST_F(TestParserClass, TestParseDataClassApi) {
  py::function fn_ = python_adapter::GetPyFn("gtest_input.pipeline.parse.parser_test", "test_class_fn");
  Parser::InitParserEnvironment(fn_);
  FuncGraphPtr func_graph = ParsePythonCode(fn_);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);

  // check the dataclass
  bool is_dataclass = false;
  py::object dataclass_obj;
  // check the dataclass
  for (auto node : manager->all_nodes()) {
    if (node->isa<ValueNode>()) {
      ValuePtr value = node->cast<ValueNodePtr>()->value();
      if (value->isa<PyObjectWrapper>()) {
        if (IsValueNode<ClassObject>(node)) {
          is_dataclass = true;
          dataclass_obj = value->cast<std::shared_ptr<PyObjectWrapper>>()->obj();
        }
      }
    }
  }

  ASSERT_TRUE(is_dataclass);

  // parse data class method
  py::object inf_method = python_adapter::GetPyObjAttr(dataclass_obj, "inf");
  FuncGraphPtr graph_inf = ParsePythonCode(inf_method);
  ASSERT_TRUE(nullptr != graph_inf);
  manager->AddFuncGraph(graph_inf);
}

/* # skip ut test cases temporarily
// Test case 2: test parse object, transfore the CELL instance to api.
TEST_F(TestParserClass, TestParseMethod) {
  py::object obj_ = python_adapter::CallPyFn("gtest_input.pipeline.parse.parse_class", "test_parse_object_instance");
  Parser::InitParserEnvironment(obj_);
  FuncGraphPtr func_graph = ParsePythonCode(obj_);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);
}

// Test case 3: common test for debug ptest case
TEST_F(TestParserClass, TestParseCompileAPI) {
  python_adapter::CallPyFn("gtest_input.pipeline.parse.parse_compile", "test_build");
  MS_LOG(DEBUG) << "Test end";
}
*/

}  // namespace parse
}  // namespace mindspore
