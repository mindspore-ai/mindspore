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
#include "c_api/include/graph.h"
#include "c_api/include/node.h"
#include "c_api/include/tensor.h"
#include "c_api/include/context.h"
#include "c_api/base/status.h"
#include "c_api/base/handle_types.h"
#include "c_api/include/attribute.h"

class TestNestedIf : public ST::Common {
 public:
  TestNestedIf() {}
};

/*
 * the pseudo-code to be implemented
 * x = 97
 * a = 4
 * if (x > 0) {
 *   x += a
 *   if (x > 100) {
 *     return 1
 *   }
 * return 0
 * }
 */
namespace {
GraphHandle BuildNestedTrueGraph(ResMgrHandle res_mgr, NodeHandle true_ret) {
  GraphHandle sub_fg_true = MSFuncGraphCreate(res_mgr);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_true, true_ret, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_true;
}

GraphHandle BuildNestedFalseGraph(ResMgrHandle res_mgr, NodeHandle false_ret) {
  GraphHandle sub_fg_false = MSFuncGraphCreate(res_mgr);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_false, false_ret, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_false;
}

GraphHandle BuildTrueGraph(ResMgrHandle res_mgr, NodeHandle x, NodeHandle a, NodeHandle true_ret,
                           NodeHandle false_ret) {
  GraphHandle sub_fg_true = MSFuncGraphCreate(res_mgr);
  NodeHandle input_nodes_1[] = {x, a};
  NodeHandle new_x = MSNewOp(res_mgr, sub_fg_true, "Add", input_nodes_1, 2, NULL, NULL, 0);
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 100);
  NodeHandle input_nodes_2[] = {new_x, n};
  NodeHandle cond = MSNewOp(res_mgr, sub_fg_true, "Greater", input_nodes_2, 2, NULL, NULL, 0);
  GraphHandle nested_fg_true = BuildNestedTrueGraph(res_mgr, true_ret);
  GraphHandle nested_fg_false = BuildNestedFalseGraph(res_mgr, false_ret);
  NodeHandle switch_res = MSNewSwitch(res_mgr, sub_fg_true, cond, nested_fg_true, nested_fg_false);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_true, switch_res, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_true;
}

GraphHandle BuildFalseGraph(ResMgrHandle res_mgr, NodeHandle false_ret) {
  GraphHandle sub_fg_false = MSFuncGraphCreate(res_mgr);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_false, false_ret, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_false;
}
}

/// Feature: C_API Control Flow
/// Description: test nested If case.
/// Expectation: case works correctly.
TEST_F(TestNestedIf, TestNestedIf) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();

  // main graph
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  NodeHandle x = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle m = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(m != nullptr);
  NodeHandle true_ret = MSNewScalarConstantInt32(res_mgr, 1);
  ASSERT_TRUE(true_ret != nullptr);
  NodeHandle false_ret = MSNewScalarConstantInt32(res_mgr, 0);
  ASSERT_TRUE(false_ret != nullptr);
  NodeHandle input_nodes_1[] = {x, m};
  NodeHandle cond = MSNewOp(res_mgr, fg, "Greater", input_nodes_1, 2, NULL, NULL, 0);
  ASSERT_TRUE(cond != nullptr);
  GraphHandle sub_fg_true = BuildTrueGraph(res_mgr, x, m, true_ret, false_ret);
  ASSERT_TRUE(sub_fg_true != nullptr);
  GraphHandle sub_fg_false = BuildFalseGraph(res_mgr, false_ret);
  ASSERT_TRUE(sub_fg_false != nullptr);
  NodeHandle switch_res = MSNewSwitch(res_mgr, fg, cond, sub_fg_true, sub_fg_false);
  ASSERT_TRUE(switch_res != nullptr);

  ret = MSFuncGraphSetOutput(res_mgr, fg, switch_res, false);
  ASSERT_TRUE(ret == RET_OK);

  // test basic funcGraph compiling and executing
  ret = MSFuncGraphCompile(res_mgr, fg);
  ASSERT_TRUE(ret == RET_OK);

  int64_t a[1] = {97};
  int64_t a_shape[1] = {1};
  int64_t b[1] = {4};
  int64_t b_shape[1] = {1};
  TensorHandle tensor_a = MSNewTensor(res_mgr, a, MS_INT32, a_shape, 1, 1 * sizeof(int));
  ASSERT_TRUE(tensor_a != nullptr);
  TensorHandle tensor_b = MSNewTensor(res_mgr, b, MS_INT32, b_shape, 1, 1 * sizeof(int));
  ASSERT_TRUE(tensor_b != nullptr);
  TensorHandle inputs[2] = {tensor_a, tensor_b};
  TensorHandle outputs[1];
  ret = MSFuncGraphRun(res_mgr, fg, inputs, 2, outputs, 1);
  ASSERT_TRUE(ret == RET_OK);
  void *data = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data != nullptr);
  ASSERT_EQ(((int *)data)[0], 1);
  MSResourceManagerDestroy(res_mgr);
}