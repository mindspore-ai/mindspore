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
#include <algorithm>

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/manager.h"
#include "utils/log_adapter.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/parse/parse.h"
#include "utils/graph_utils.h"
#include "pipeline/resource.h"
#include "debug/draw.h"
#include "operator/ops.h"
#include "vm/segment_runner.h"
#include "vm/transform.h"
#include "ir/tensor.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace compile {
using Tensor = tensor::Tensor;

class TestCompileSegmentRunner : public UT::Common {
 public:
  TestCompileSegmentRunner() : get_py_fun_("gtest_input.vm", true) { UT::InitPythonPath(); }

 protected:
  UT::PyFuncGraphFetcher get_py_fun_;
  VM vm_;
};

TEST_F(TestCompileSegmentRunner, test_MsVmConvert1) {
  FuncGraphPtr g = get_py_fun_("scalar_add");
  // g was managed by local variable manager in get_py_fun_ and that manager will be freed as no reference.
  // so a new manager should be declared to make get_outputs() in segment_runner.cc happy.
  std::shared_ptr<mindspore::FuncGraphManager> manager = mindspore::Manage(g);

  BackendPtr b = std::make_shared<Backend>("vm");
  CompileGraph transform_(b);
  auto splits = transform_.SplitNodes(g);
  VectorRef args({1.0, 2.0});

  std::vector<BaseRef> todos(splits.size());
  auto it = std::copy_if(std::begin(splits), std::end(splits), std::begin(todos),
                         [](const BaseRef& seg) -> bool { return utils::isa<VectorRef>(seg); });
  todos.resize(std::distance(todos.begin(), it));
  ASSERT_EQ(todos.size(), 1);

  AnfNodePtrList anf_list; 
  for (auto &item : utils::cast<VectorRef>(todos[0])) {
    anf_list.push_back(utils::cast<AnfNodePtr>(item));
  }
  auto convertResult = MsVmConvert(anf_list, "");
  auto runResult = (*(convertResult.run))(args);
  ASSERT_TRUE(runResult.size() == 1 && py::cast<double>(BaseRefToPyData(runResult[0])) == 3.0);
}

TEST_F(TestCompileSegmentRunner, test_MsVmConvert2) {
  FuncGraphPtr g = get_py_fun_("scalar_mul");
  std::shared_ptr<mindspore::FuncGraphManager> manager = mindspore::Manage(g);

  BackendPtr b = std::make_shared<Backend>("vm");
  CompileGraph transform_(b);
  auto splits = transform_.SplitNodes(g);
  VectorRef args({1.0, 2.0});

  std::vector<BaseRef> todos(splits.size());
  auto it = std::copy_if(std::begin(splits), std::end(splits), std::begin(todos),
                         [](const BaseRef& seg) -> bool { return utils::isa<VectorRef>(seg); });
  todos.resize(std::distance(todos.begin(), it));
  ASSERT_EQ(todos.size(), 1);

  AnfNodePtrList anf_list; 
  for (auto &item : utils::cast<VectorRef>(todos[0])) {
    anf_list.push_back(utils::cast<AnfNodePtr>(item));
  }
  auto convertResult = MsVmConvert(anf_list, "");
  auto runResult = (*(convertResult.run))(args);
  ASSERT_TRUE(runResult.size() == 1 && py::cast<double>(BaseRefToPyData(runResult[0])) == 2.0);
}

TEST_F(TestCompileSegmentRunner, test_if) {
  FuncGraphPtr g = get_py_fun_("test_if");
  std::shared_ptr<mindspore::FuncGraphManager> manager = mindspore::Manage(g);

  BackendPtr b = std::make_shared<Backend>("vm");
  CompileGraph transform_(b);
  auto splits = transform_.SplitNodes(g);
  VectorRef args({1.0, 2.0});

  std::vector<BaseRef> todos(splits.size());
  auto it = std::copy_if(std::begin(splits), std::end(splits), std::begin(todos),
                         [](const BaseRef& seg) -> bool { return utils::isa<VectorRef>(seg); });
  todos.resize(std::distance(todos.begin(), it));
  ASSERT_EQ(todos.size(), 1);

  AnfNodePtrList anf_list; 
  for (auto &item : utils::cast<VectorRef>(todos[0])) {
    anf_list.push_back(utils::cast<AnfNodePtr>(item));
  }
  auto convertResult = MsVmConvert(anf_list, "");
  auto runResult = (*(convertResult.run))(args);

  auto result = py::cast<bool>(BaseRefToPyData(runResult[0]));
  ASSERT_TRUE(runResult.size() == 1 && result == false);
}

TEST_F(TestCompileSegmentRunner, test_RunOperation1) {
  VectorRef args({1});
  auto res = RunOperation(prim::kPrimIdentity, args);
  ASSERT_EQ(py::cast<int>(BaseRefToPyData(res)), 1);
}

TEST_F(TestCompileSegmentRunner, test_RunOperation2) {
  VectorRef args({1, 2});
  auto res = RunOperation(prim::kPrimScalarGt, args);
  ASSERT_EQ(py::cast<bool>(BaseRefToPyData(res)), false);
}
}  // namespace compile
}  // namespace mindspore
