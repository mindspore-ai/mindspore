/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include "common/common_test.h"
#include "include/c_api/ms/graph.h"
#include "include/c_api/ms/node.h"
#include "include/c_api/ms/tensor.h"
#include "include/c_api/ms/context.h"
#include "include/c_api/ms/base/status.h"
#include "include/c_api/ms/base/handle_types.h"
#include "include/c_api/ms/value.h"

class TestIRPass : public ST::Common {
 public:
  TestIRPass() {}
};

/// Feature: C_API Graph
/// Description: test ir pass case.
/// Expectation: case works correctly.
TEST_F(TestIRPass, TestAutoMonadElimPass) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();

  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(res_mgr != nullptr);
  NodeHandle x = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle y = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(y != nullptr);
  NodeHandle para_1 = MSNewVariableScalarInt32(res_mgr, fg, 1);
  ASSERT_TRUE(para_1 != nullptr);
  NodeHandle para_2 = MSNewVariableScalarInt32(res_mgr, fg, 3);
  ASSERT_TRUE(para_2 != nullptr);

  const char *assign_attr_name[] = {"side_effect_mem"};
  ValueHandle side_effect_mem = MSNewValueBool(res_mgr, true);
  ASSERT_TRUE(side_effect_mem != nullptr);
  ValueHandle assign_attrs[] = {side_effect_mem};
  const char *depend_attr_name[] = {"side_effect_propagate"};
  ValueHandle side_effect_propagate = MSNewValueInt64(res_mgr, 1);
  ASSERT_TRUE(side_effect_propagate != nullptr);
  ValueHandle depend_attrs[] = {side_effect_propagate};

  NodeHandle input_nodes_1[] = {para_1, x};
  NodeHandle assign_1 = MSNewOp(res_mgr, fg, "Assign", input_nodes_1, 2, assign_attr_name, assign_attrs, 1);
  ASSERT_TRUE(assign_1 != nullptr);
  NodeHandle input_nodes_2[] = {para_2, y};
  NodeHandle assign_2 = MSNewOp(res_mgr, fg, "Assign", input_nodes_2, 2, assign_attr_name, assign_attrs, 1);
  ASSERT_TRUE(assign_2 != nullptr);
  NodeHandle input_nodes_3[] = {assign_1, assign_2};
  NodeHandle make_tuple_1 = MSPackNodesTuple(res_mgr, fg, input_nodes_3, 2);
  ASSERT_TRUE(make_tuple_1 != nullptr);
  NodeHandle input_nodes_4[] = {make_tuple_1};
  NodeHandle stop_grad_1 = MSNewOp(res_mgr, fg, "StopGradient", input_nodes_4, 1, NULL, NULL, 0);
  ASSERT_TRUE(stop_grad_1 != nullptr);
  NodeHandle input_nodes_5[] = {para_2, stop_grad_1};
  NodeHandle depend_1 = MSNewOp(res_mgr, fg, "Depend", input_nodes_5, 2, depend_attr_name, depend_attrs, 1);
  ASSERT_TRUE(depend_1 != nullptr);

  ret = MSFuncGraphSetOutput(res_mgr, fg, depend_1, false);
  ASSERT_TRUE(ret == RET_OK);

  // test basic funcGraph compiling and executing
  OptPassID passes[] = {MS_AUTO_MONAD_ELIM_PASS};
  ret = MSFuncGraphCompile(res_mgr, fg, passes, 1);
  ASSERT_TRUE(ret == RET_OK);
  TensorHandle a = MSNewTensorScalarInt32(res_mgr, 10);
  ASSERT_TRUE(a != nullptr);
  TensorHandle b = MSNewTensorScalarInt32(res_mgr, 30);
  ASSERT_TRUE(b != nullptr);
  TensorHandle inputs[2] = {a, b};
  TensorHandle outputs[1];
  ret = MSFuncGraphRun(res_mgr, fg, inputs, 2, outputs, 1);
  ASSERT_TRUE(ret == RET_OK);
  void *data = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data != nullptr);
  ASSERT_EQ(((int *)data)[0], 30);

  MSResourceManagerDestroy(res_mgr);
}