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

  // draw graph
  int i = 0;
  for (auto func_graph : manager->func_graphs()) {
    std::string name = "ut_resolve_graph_" + std::to_string(i) + ".dot";
    draw::Draw(name, func_graph);
    i++;
  }
}

TEST_F(TestResolve, TestParseGraphTestClosureResolve) {
  py::function test_fn =
    python_adapter::CallPyFn("gtest_input.pipeline.parse.parser_test", "test_reslove_closure", 123);
  FuncGraphPtr func_graph = ParsePythonCode(test_fn);
  ASSERT_TRUE(func_graph != nullptr);
  draw::Draw("test_reslove_closure.dot", func_graph);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);

  ASSERT_EQ(manager->func_graphs().size(), (size_t)2);

  // draw graph
  int i = 0;
  for (auto func_graph : manager->func_graphs()) {
    std::string name = "ut_test_reslove_closure_graph_" + std::to_string(i) + ".dot";
    draw::Draw(name, func_graph);
    i++;
  }
}
}  // namespace parse
}  // namespace mindspore
