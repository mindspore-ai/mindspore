/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <string>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/convert_utils.h"
#include "ops/arithmetic_ops.h"

namespace mindspore {
class TestFuncGraphBuilder : public UT::Common {
 public:
  TestFuncGraphBuilder() : get_py_fun_("gtest_input.pipeline.pi.func_graph_builder", true) {}

  virtual void SetUp();

  virtual void TearDown();

  bool CheckEqual(const FuncGraphPtr &fg1, const FuncGraphPtr &fg2) {
    equiv_graph_.clear();
    equiv_node_.clear();
    return Isomorphic(fg1, fg2, &equiv_graph_, &equiv_node_);
  }

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
  FuncGraphPairMapEquiv equiv_graph_;
  NodeMapEquiv equiv_node_;
};

void TestFuncGraphBuilder::SetUp() {}

void TestFuncGraphBuilder::TearDown() {}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add inputs and add outputs.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, TestAddInputAddOutput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto input1 = func_graph_builder.AddInput(int_v1);
  EXPECT_NE(input1.ptr(), nullptr);
  py::int_ int_v2 = 2;
  auto input2 = func_graph_builder.AddInput(int_v2);
  EXPECT_NE(input2.ptr(), nullptr);
  EXPECT_TRUE(func_graph_builder.AddOutput(input2));
  auto graph = func_graph_builder.graph();
  EXPECT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_inputs_and_outputs", "graph");
  EXPECT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, TestAddNode) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto input1 = func_graph_builder.AddInput(int_v1);
  EXPECT_NE(input1.ptr(), nullptr);
  py::int_ int_v2 = 2;
  auto input2 = func_graph_builder.AddInput(int_v2);
  EXPECT_NE(input2.ptr(), nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  EXPECT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  EXPECT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj = func_graph_builder.AddNode(scalar_add_prim, {input1, input2});
  EXPECT_NE(obj.ptr(), nullptr);
  EXPECT_TRUE(func_graph_builder.AddOutput(obj));
  auto graph = func_graph_builder.graph();
  EXPECT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_node", "graph");
  EXPECT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode with unknown input.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, TestAddNodeUnkownInput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto input1 = func_graph_builder.AddInput(int_v1);
  EXPECT_NE(input1.ptr(), nullptr);
  py::int_ int_v2 = 2;
  auto obj = func_graph_builder.AddNode(prim::kPrimScalarAdd, {input1, int_v2});
  EXPECT_EQ(obj.ptr(), nullptr);
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add func_graph called node.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, TestAddFgCallNode) {
  FuncGraphBuilder func_graph_builder1;
  py::int_ int_v1 = 1;
  auto input1 = func_graph_builder1.AddInput(int_v1);
  EXPECT_NE(input1.ptr(), nullptr);
  py::int_ int_v2 = 2;
  auto input2 = func_graph_builder1.AddInput(int_v2);
  EXPECT_NE(input2.ptr(), nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  EXPECT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  EXPECT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj = func_graph_builder1.AddNode(scalar_add_prim, {input1, input2});
  EXPECT_NE(obj.ptr(), nullptr);
  EXPECT_TRUE(func_graph_builder1.AddOutput(obj));
  auto graph1 = func_graph_builder1.graph();
  EXPECT_NE(graph1, nullptr);

  FuncGraphBuilder func_graph_builder2;
  input1 = func_graph_builder2.AddInput(int_v1);
  EXPECT_NE(input1.ptr(), nullptr);
  input2 = func_graph_builder2.AddInput(int_v2);
  EXPECT_NE(input2.ptr(), nullptr);
  auto call_node_obj = func_graph_builder2.AddNode(graph1, {input1, input2});
  EXPECT_NE(call_node_obj.ptr(), nullptr);
  EXPECT_TRUE(func_graph_builder2.AddOutput(call_node_obj));
  auto graph2 = func_graph_builder2.graph();
  DumpIR("graph2.ir", graph2);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_fg_call_node", "graph");
  EXPECT_TRUE(CheckEqual(graph2, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to get the function of a method.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, TestGetFunctionFromMethod) {}
}  // namespace mindspore
