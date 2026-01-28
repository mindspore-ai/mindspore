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
#include "frontend/jit/ps/parse/parse.h"
#include "frontend/jit/ps/resource.h"
#include "include/utils/ir_dump/draw_interface.h"

namespace mindspore {
namespace parse {

class TestResolve : public UT::Common {
 public:
  TestResolve() {}
  virtual void SetUp();
  virtual void TearDown();
};

void TestResolve::SetUp() { UT::InitPythonPath(); }

void TestResolve::TearDown() {}

TEST_F(TestResolve, TestResolveApi) {
  py::function fn_ = python_adapter::GetPyFn("gtest_input.pipeline.parse.parser_test", "get_resolve_fn");

  // parse graph
  FuncGraphPtr func_graph = ParsePythonCode(fn_);
  ASSERT_FALSE(nullptr == func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);

  ASSERT_EQ(manager->func_graphs().size(), (size_t)2);
}

TEST_F(TestResolve, TestParseGraphTestClosureResolve) {
  py::function test_fn =
    python_adapter::CallPyFn("gtest_input.pipeline.parse.parser_test", "test_reslove_closure", 123);
  FuncGraphPtr func_graph = ParsePythonCode(test_fn);
  ASSERT_TRUE(func_graph != nullptr);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);

  ASSERT_EQ(manager->func_graphs().size(), (size_t)2);
}

// Feature: Resolve.
// Description: Parse the graph with Parameter.
// Expectation: The Parameter is added to the top_graph.
TEST_F(TestResolve, TestResolveTopGraph) {
  py::function fn_ = python_adapter::GetPyFn("gtest_input.pipeline.parse.parser_test", "get_resolve_fn_with_parameter");

  // parse graph
  FuncGraphPtr func_graph = ParsePythonCode(fn_);
  ASSERT_FALSE(nullptr == func_graph);

  // call resolve
  auto top_graph = std::make_shared<FuncGraph>();
  auto res = std::make_shared<pipeline::Resource>();
  res->set_func_graph(top_graph);
  auto ret = parse::ResolveFuncGraph(func_graph, res, false);
  ASSERT_TRUE(ret);
  ASSERT_EQ(top_graph->parameters().size(), 1);
}
}  // namespace parse
}  // namespace mindspore
