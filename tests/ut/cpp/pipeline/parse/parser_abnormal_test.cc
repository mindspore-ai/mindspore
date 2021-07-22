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
#include "utils/profile.h"
#include "pipeline/jit/parse/parse.h"
#include "debug/draw.h"

namespace mindspore {
namespace parse {

class TestParserAbnormal : public UT::Common {
 public:
  TestParserAbnormal() : getPyFun("gtest_input.pipeline.parse.parse_abnormal") {}
  virtual void SetUp();
  virtual void TearDown();

 public:
  UT::PyFuncGraphFetcher getPyFun;
};

void TestParserAbnormal::SetUp() { UT::InitPythonPath(); }

void TestParserAbnormal::TearDown() {}

TEST_F(TestParserAbnormal, TestParseRecursion) {
  FuncGraphPtr func_graph = getPyFun("test_keep_roots_recursion");
  ASSERT_TRUE(nullptr != func_graph);

  // save the func func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);
}

int test_performance(int x) { return x; }

TEST_F(TestParserAbnormal, TestPythonAdapterPerformance) {
  MS_LOG(INFO) << "TestPythonAdapterPerformance start";
  std::shared_ptr<py::scoped_interpreter> env = python_adapter::set_python_scoped();
  py::module mod = python_adapter::GetPyModule("gtest_input.pipeline.parse.parse_abnormal");

  // call the python function
  std::size_t count = 1000000;
  double t1 = GetTime();
  for (std::size_t i = 0; i < count; i++) {
    mod.attr("test_performance")(i);
  }
  double t2 = GetTime();
  printf("Call python function %lu time is : %f", count, t2 - t1);

  // call the python function
  t1 = GetTime();
  for (std::size_t i = 0; i < count; i++) {
    test_performance(i);
  }
  t2 = GetTime();
  printf("Call c++ function %lu time is : %f", count, t2 - t1);
}

// test the single Expr statement
TEST_F(TestParserAbnormal, TestParseExprStatement) {
  FuncGraphPtr func_graph = getPyFun("test_list_append");
  ASSERT_TRUE(nullptr != func_graph);

  // save the func func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);

  ASSERT_TRUE(ret_);

  // check the 'append' node
  bool is_append_node = false;
  int count = 0;
  py::object dataclass_obj;
  // check the dataclass
  for (auto node : manager->all_nodes()) {
    if (node != nullptr && node->isa<ValueNode>() && node->cast<ValueNodePtr>()->value()->isa<StringImm>()) {
      if (GetValue<std::string>(node->cast<ValueNodePtr>()->value()) == "append") {
        is_append_node = true;
        count++;
      }
    }
  }
  ASSERT_TRUE(is_append_node);
  ASSERT_EQ(count, 2);
  MS_LOG(INFO) << "append node have: " << count << " .";
}

}  // namespace parse
}  // namespace mindspore
