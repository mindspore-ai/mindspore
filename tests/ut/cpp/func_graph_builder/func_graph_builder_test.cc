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
#include "pipeline/jit/pi/graph_capture/abstract_wrapper.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/convert_utils.h"
#include "ops/arithmetic_ops.h"
#include "ops/other_ops.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
class TestFuncGraphBuilder : public UT::Common {
 public:
  TestFuncGraphBuilder() : get_py_fun_("gtest_input.pipeline.pi.func_graph_builder", true) {}

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

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add inputs and add outputs.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, TestAddInputAddOutput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto input1 = func_graph_builder.AddTopGraphArgInput(int_v1);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto input2 = func_graph_builder.AddTopGraphArgInput(int_v2);
  ASSERT_NE(input2, nullptr);
  ASSERT_TRUE(func_graph_builder.AddOutput(input2));
  auto graph = func_graph_builder.graph();
  ASSERT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_inputs_and_outputs", "graph");
  ASSERT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddNodeAndSingleOutput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj = func_graph_builder.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj, nullptr);
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  auto graph = func_graph_builder.graph();
  ASSERT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_node", "graph_single_output");
  ASSERT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddNodeAndMultiOutput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj = func_graph_builder.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj, nullptr);
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  auto graph = func_graph_builder.graph();
  ASSERT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_node", "graph_multi_output");
  ASSERT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode with constant input.
// Expectation: Failed to add the node.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddNodeConstantInput) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto obj = func_graph_builder.AddNode(prim::kPrimScalarAdd, {input1, v2_wrapper});
  ASSERT_NE(obj, nullptr);
  ASSERT_TRUE(func_graph_builder.AddOutput(obj));
  auto graph = func_graph_builder.graph();
  ASSERT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_node_with_constant", "graph");
  ASSERT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode with an uncallable object.
// Expectation: Failed to add the node.
TEST_F(TestFuncGraphBuilder, TestAddNodeUnCallable) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto obj = func_graph_builder.AddNode(scalar_add_prim_class, {input1, input2});
  ASSERT_EQ(obj, nullptr);
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add cnode with constant input.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddMultiNode) {
  FuncGraphBuilder func_graph_builder;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto add_obj = func_graph_builder.AddMultiNode("add", {input1, input2});
  ASSERT_TRUE(func_graph_builder.AddOutput(add_obj));
  auto graph = func_graph_builder.graph();
  ASSERT_NE(graph, nullptr);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_binary_node", "graph");
  ASSERT_TRUE(CheckEqual(graph, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add func_graph called node.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddFgCallNodeSingleOutput) {
  FuncGraphBuilder func_graph_builder1;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder1.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder1.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder1.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder1.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj = func_graph_builder1.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj, nullptr);
  ASSERT_TRUE(func_graph_builder1.AddOutput(obj));
  auto graph1 = func_graph_builder1.graph();
  ASSERT_NE(graph1, nullptr);

  FuncGraphBuilder func_graph_builder2;
  v1_wrapper = func_graph_builder2.AddLocalVariable(int_v1);
  input1 = func_graph_builder2.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  v2_wrapper = func_graph_builder2.AddLocalVariable(int_v2);
  input2 = func_graph_builder2.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto call_node_obj = func_graph_builder2.AddNode(graph1, {input1, input2});
  ASSERT_NE(call_node_obj, nullptr);
  ASSERT_TRUE(func_graph_builder2.AddOutput(call_node_obj));
  auto graph2 = func_graph_builder2.graph();
  DumpIR("graph2.ir", graph2);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_fg_call_node", "graph_single_output");
  ASSERT_TRUE(CheckEqual(graph2, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to add func_graph called node.
// Expectation: The expected graph is constructed.
TEST_F(TestFuncGraphBuilder, DISABLED_TestAddFgCallNodeMultiOutput) {
  FuncGraphBuilder func_graph_builder1;
  py::int_ int_v1 = 1;
  auto v1_wrapper = func_graph_builder1.AddLocalVariable(int_v1);
  auto input1 = func_graph_builder1.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  py::int_ int_v2 = 2;
  auto v2_wrapper = func_graph_builder1.AddLocalVariable(int_v2);
  auto input2 = func_graph_builder1.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto mod = python_adapter::GetPyModule("mindspore.ops.operations._scalar_ops");
  ASSERT_FALSE(py::isinstance<py::none>(mod));
  auto scalar_add_prim_class = mod.attr("ScalarAdd");
  ASSERT_FALSE(py::isinstance<py::none>(scalar_add_prim_class));
  auto scalar_add_prim = scalar_add_prim_class();
  auto obj1 = func_graph_builder1.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj1, nullptr);
  ASSERT_TRUE(func_graph_builder1.AddOutput(obj1));
  auto obj2 = func_graph_builder1.AddNode(scalar_add_prim, {input1, input2});
  ASSERT_NE(obj2, nullptr);
  ASSERT_TRUE(func_graph_builder1.AddOutput(obj2));
  auto graph1 = func_graph_builder1.graph();
  ASSERT_NE(graph1, nullptr);

  FuncGraphBuilder func_graph_builder2;
  v1_wrapper = func_graph_builder2.AddLocalVariable(int_v1);
  input1 = func_graph_builder2.AddSubGraphInput(v1_wrapper);
  ASSERT_NE(input1, nullptr);
  v2_wrapper = func_graph_builder2.AddLocalVariable(int_v2);
  input2 = func_graph_builder2.AddSubGraphInput(v2_wrapper);
  ASSERT_NE(input2, nullptr);
  auto call_node_obj = func_graph_builder2.AddNode(graph1, {input1, input2});
  ASSERT_NE(call_node_obj, nullptr);
  ASSERT_TRUE(func_graph_builder2.AddOutput(call_node_obj));
  auto graph2 = func_graph_builder2.graph();
  DumpIR("graph2.ir", graph2);
  FuncGraphPtr expected_graph = get_py_fun_.CallAndParseRet("test_add_fg_call_node", "graph_multi_output");
  ASSERT_TRUE(CheckEqual(graph2, expected_graph));
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to get the function or primitive from a method.
// Expectation: Get the correct function or primitive.
TEST_F(TestFuncGraphBuilder, DISABLED_TestGetFunctionFromMethod) {
  py::tuple t;
  auto func = FuncGraphBuilder::ConvertMethod(t.attr("index"));
  ASSERT_NE(func.ptr(), nullptr);
  ASSERT_EQ(func.attr("__name__").cast<std::string>(), "sequence_index");

  func = FuncGraphBuilder::ConvertMethod(t.attr("__getitem__"));
  ASSERT_NE(func.ptr(), nullptr);
  ASSERT_EQ(func.attr("name").cast<std::string>(), prim::kPrimTupleGetItem->name());
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to get the callable obj from a function.
// Expectation: Get the correct callable obj.
TEST_F(TestFuncGraphBuilder, TestGetCallableObjFromFunction) {
  auto operator_mod = python_adapter::GetPyModule("operator");
  auto func_add = python_adapter::GetPyObjAttr(operator_mod, "add");
  auto callable_obj = FuncGraphBuilder::ConvertFunction(func_add);
  ASSERT_NE(callable_obj.ptr(), nullptr);
  ASSERT_TRUE(py::isinstance<MetaFuncGraph>(callable_obj));

  auto builtin_mod = python_adapter::GetPyModule("builtins");
  auto func_abs = python_adapter::GetPyObjAttr(builtin_mod, "abs");
  callable_obj = FuncGraphBuilder::ConvertFunction(func_abs);
  ASSERT_NE(callable_obj.ptr(), nullptr);
  ASSERT_EQ(callable_obj.attr("name").cast<std::string>(), prim::kPrimInnerAbs->name());
}

// Feature: Build graph in pi_jit.
// Description: Use the func_graph_builder api to check if an obj can be constantly folded.
// Expectation: Get the correct result.
TEST_F(TestFuncGraphBuilder, TestCanConstantFoldFunc) {
  auto operator_mod = python_adapter::GetPyModule("operator");
  auto func_add = python_adapter::GetPyObjAttr(operator_mod, "add");
  ASSERT_TRUE(FuncGraphBuilder::CanConstantFoldFunc(func_add));

  auto builtin_mod = python_adapter::GetPyModule("builtins");
  auto func_abs = python_adapter::GetPyObjAttr(builtin_mod, "abs");
  ASSERT_TRUE(FuncGraphBuilder::CanConstantFoldFunc(func_abs));

  auto ms_mod = python_adapter::GetPyModule("mindspore");
  auto func_ms_memory_recycle = python_adapter::GetPyObjAttr(builtin_mod, "ms_memory_recycle");
  ASSERT_FALSE(FuncGraphBuilder::CanConstantFoldFunc(func_ms_memory_recycle));
}
}  // namespace mindspore
