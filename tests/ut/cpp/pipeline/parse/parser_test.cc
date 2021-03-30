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
class TestParser : public UT::Common {
 public:
  TestParser() {}
  virtual void SetUp();
  virtual void TearDown();

  py::function fn;

  py::function GetPythonFunction(std::string function);
};

void TestParser::SetUp() { UT::InitPythonPath(); }

void TestParser::TearDown() {}

py::function TestParser::GetPythonFunction(std::string function) {
  // init resource
  try {
    fn = python_adapter::GetPyFn("gtest_input.pipeline.parse.parser_test", function.c_str());
    return fn;
  } catch (...) {
    MS_LOG(ERROR) << "get fn failure!!!";
  }
  return py::none();
}

TEST_F(TestParser, TestParseApi) {
  // Test null fn
  py::function fn_null;
  FuncGraphPtr func_graph = ParsePythonCode(fn_null);
  ASSERT_TRUE(nullptr == func_graph);

  // Test parse api
  GetPythonFunction("test_f");
  func_graph = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != func_graph);
}

TEST_F(TestParser, TestParseAst) {
  GetPythonFunction("test_f");

  ParseAst ast = ParseAst(fn);
  bool succ = ast.InitParseAstInfo();
  ASSERT_TRUE(succ = true);

  // get FunctionDef node
  py::object node = ast.GetAstNode();

  // check arg
  std::string fun_args[] = {"x", "y"};
  std::string fun_name = "test_f";
  py::list args = ast.GetArgs(node);
  for (std::size_t i = 0; i < args.size(); i++) {
    py::str pyArg = args[i].attr("arg");
    std::string arg = pyArg;
    ASSERT_STREQ(arg.c_str(), fun_args[i].c_str());
  }

  // check function name
  // get function name
  py::str name = python_adapter::GetPyObjAttr(node, "name");
  std::string function_name = name;
  ASSERT_STREQ(function_name.c_str(), fun_name.c_str());
}

TEST_F(TestParser, TestParseGraphSuccess) {
  GetPythonFunction("test_f");
  // parse fn to graph
  FuncGraphPtr func_graph = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != func_graph);
}

TEST_F(TestParser, TestParseGraphIf) {
  GetPythonFunction("test_if");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphIfExp) {
  GetPythonFunction("test_ifexp");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphIfNested) {
  GetPythonFunction("test_if_nested");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseWhile) {
  GetPythonFunction("test_while");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphNum) {
  FuncGraphPtr ret_val;
  GetPythonFunction("testDoNum");
  ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphStr) {
  FuncGraphPtr ret_val;
  GetPythonFunction("testDoStr");
  ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphNamedConst) {
  FuncGraphPtr ret_val;
  GetPythonFunction("testDoNamedConstTrue");
  ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
  GetPythonFunction("testDoNamedConstFalse");
  ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
  GetPythonFunction("testDoNamedConstNone");
  ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphForStatement) {
  GetPythonFunction("test_for");

  FuncGraphPtr func_graph = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);

  // draw graph
  int i = 0;
  for (auto tmp : manager->func_graphs()) {
    std::string name = "ut_parser_for_loop_" + std::to_string(i) + ".dot";
    draw::Draw(name, tmp);
    i++;
  }
}

TEST_F(TestParser, TestParseGraphCompareExprLt) {
  GetPythonFunction("test_compare_lt");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphCompareExprGt) {
  GetPythonFunction("test_compare_gt");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphCompareExprLe) {
  GetPythonFunction("test_compare_le");
  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphCompareExprNe) {
  GetPythonFunction("test_compare_ne");
  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphCompareExprGe) {
  GetPythonFunction("test_compare_ge");
  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphCompareExprEq) {
  GetPythonFunction("test_compare_eq");
  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphBoolOpTwoAnd) {
  GetPythonFunction("test_boolop_two_and");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphBoolOpThreeAnd) {
  GetPythonFunction("test_boolop_three_and");
  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphBoolOpTwoOr) {
  GetPythonFunction("test_boolop_two_or");
  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphBoolOpThreeOr) {
  GetPythonFunction("test_boolop_three_or");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphBoolOpMixAndOr) {
  GetPythonFunction("test_boolop_mix_and_or");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphLambda) {
  GetPythonFunction("test_lambda");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphFuncDef) {
  GetPythonFunction("test_funcdef");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphSimpleClosure) {
  GetPythonFunction("test_simple_closure");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphTestTuple) {
  GetPythonFunction("test_tuple_fn");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphTupleAssign) {
  GetPythonFunction("test_assign_tuple");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphTestList) {
  GetPythonFunction("test_list_fn");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphUnaryOp) {
  GetPythonFunction("test_unary");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphAguassign) {
  GetPythonFunction("test_augassign");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseSystemFunction) {
  GetPythonFunction("test_sys_call");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);
}

TEST_F(TestParser, TestParseGraphBoolNot) {
  GetPythonFunction("test_bool_not");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(ret_val);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);

  // draw graph
  int i = 0;
  for (auto tmp : manager->func_graphs()) {
    std::string name = "ut_parser_for_not_" + std::to_string(i) + ".dot";
    draw::Draw(name, tmp);
    i++;
  }
}

TEST_F(TestParser, TestCallPythonFnUseTupleParamete) {
  GetPythonFunction("test_call_fn_use_tuple");

  py::tuple params = py::tuple(5);
  params[0] = 0;
  params[1] = 1;
  params[2] = 2.0;
  params[3] = fn;
  params[4] = "test_call_fn_use_tuple";
  py::object result =
    python_adapter::CallPyFn("gtest_input.pipeline.parse.parser_test", "test_call_fn_use_tuple", params);

  int ret_size = py::cast<int>(result);

  ASSERT_EQ(ret_size, 5);
}

TEST_F(TestParser, TestParseGraphSubscriptSetitem) {
  GetPythonFunction("test_subscript_setitem");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);

  std::shared_ptr<FuncGraphManager> manager = Manage(ret_val);
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);
}

TEST_F(TestParser, TestParseGraphDict) {
  GetPythonFunction("test_dict");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);

  std::shared_ptr<FuncGraphManager> manager = Manage(ret_val);
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);
}

TEST_F(TestParser, TestParseGraphCallVargs) {
  GetPythonFunction("test_call_variable");

  FuncGraphPtr ret_val = ParsePythonCode(fn);
  ASSERT_TRUE(nullptr != ret_val);

  std::shared_ptr<FuncGraphManager> manager = Manage(ret_val);
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);
}

TEST_F(TestParser, TestParserUndefinedVar) {
  py::function fn_ = python_adapter::GetPyFn("gtest_input.pipeline.parse.parser_test", "test_parse_undefined_var");

  // parse undefined var
  EXPECT_THROW({ ParsePythonCode(fn_); }, std::runtime_error);
}
}  // namespace parse
}  // namespace mindspore
