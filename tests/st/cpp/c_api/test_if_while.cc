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
#include "c_api/base/status.h"
#include "c_api/base/handle_types.h"
#include "c_api/include/attribute.h"

class TestIfWhile : public ST::Common {
 public:
  TestIfWhile() {}
};

/*
 * the pseudo-code to be implemented
 * x = 1
 * if (x < 10) {
 *   for (int i=0; i<10; ++i) {
 *     x *= 2
 *   }
 * } else {
 *   for (int i=0; i<10; ++i) {
 *     x += 2
 *   }
 * }
 * return x
 */
namespace {
GraphHandle BuildNestedCondGraphTrue(ResMgrHandle res_mgr, NodeHandle i) {
  GraphHandle sub_fg_cond = MSFuncGraphCreate(res_mgr);
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 10);
  NodeHandle input_nodes[] = {i, n};
  NodeHandle cond = MSNewOp(res_mgr, sub_fg_cond, "Less", input_nodes, 2, NULL, NULL, 0);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_cond, cond, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_cond;
}

GraphHandle BuildNestedBodyGraphTrue(ResMgrHandle res_mgr, NodeHandle x, NodeHandle i) {
  GraphHandle sub_fg_body = MSFuncGraphCreate(res_mgr);
  NodeHandle step1 = MSNewScalarConstantInt32(res_mgr, 2);
  NodeHandle step2 = MSNewScalarConstantInt32(res_mgr, 1);
  NodeHandle input_nodes_body[] = {x, step1};
  NodeHandle new_x = MSNewOp(res_mgr, sub_fg_body, "Mul", input_nodes_body, 2, NULL, NULL, 0);
  NodeHandle input_nodes_i[] = {i, step2};
  NodeHandle new_i = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_i, 2, NULL, NULL, 0);
  NodeHandle out_nodes[2] = {new_x, new_i};
  STATUS ret = MSFuncGraphSetOutputs(res_mgr, sub_fg_body, out_nodes, 2, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_body;
}

GraphHandle BuildNestedAfterGraphTrue(ResMgrHandle res_mgr, NodeHandle x) {
  GraphHandle sub_fg_after = MSFuncGraphCreate(res_mgr);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_after, x, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_after;
}

GraphHandle BuildNestedCondGraphFalse(ResMgrHandle res_mgr, NodeHandle i) {
  GraphHandle sub_fg_cond = MSFuncGraphCreate(res_mgr);
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 10);
  NodeHandle input_nodes[] = {i, n};
  NodeHandle cond = MSNewOp(res_mgr, sub_fg_cond, "Less", input_nodes, 2, NULL, NULL, 0);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_cond, cond, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_cond;
}

GraphHandle BuildNestedBodyGraphFalse(ResMgrHandle res_mgr, NodeHandle x, NodeHandle i) {
  GraphHandle sub_fg_body = MSFuncGraphCreate(res_mgr);
  NodeHandle step1 = MSNewScalarConstantInt32(res_mgr, 2);
  NodeHandle step2 = MSNewScalarConstantInt32(res_mgr, 1);
  NodeHandle input_nodes_body[] = {x, step1};
  NodeHandle new_x = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_body, 2, NULL, NULL, 0);
  NodeHandle input_nodes_i[] = {i, step2};
  NodeHandle new_i = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_i, 2, NULL, NULL, 0);
  NodeHandle out_nodes[2] = {new_x, new_i};
  STATUS ret = MSFuncGraphSetOutputs(res_mgr, sub_fg_body, out_nodes, 2, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_body;
}

GraphHandle BuildNestedAfterGraphFalse(ResMgrHandle res_mgr, NodeHandle x) {
  GraphHandle sub_fg_after = MSFuncGraphCreate(res_mgr);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_after, x, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_after;
}

GraphHandle BuildTrueGraph(ResMgrHandle res_mgr, NodeHandle x) {
  GraphHandle sub_fg_true = MSFuncGraphCreate(res_mgr);
  int data_i[] = {0};
  int64_t shape_i[] = {1};
  NodeHandle i = MSNewTensorVariable(res_mgr, sub_fg_true, data_i, MS_INT32, shape_i, 1, 1 * sizeof(int));
  GraphHandle nested_fg_cond = BuildNestedCondGraphTrue(res_mgr, i);
  GraphHandle nested_fg_body = BuildNestedBodyGraphTrue(res_mgr, x, i);
  GraphHandle nested_fg_after = BuildNestedAfterGraphTrue(res_mgr, x);
  NodeHandle new_x = MSNewWhile(res_mgr, sub_fg_true, nested_fg_cond, nested_fg_body, nested_fg_after);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_true, new_x, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_true;
}

GraphHandle BuildFalseGraph(ResMgrHandle res_mgr, NodeHandle x) {
  GraphHandle sub_fg_false = MSFuncGraphCreate(res_mgr);
  int data_i[] = {0};
  int64_t shape_i[] = {1};
  NodeHandle i = MSNewTensorVariable(res_mgr, sub_fg_false, data_i, MS_INT32, shape_i, 1, 1 * sizeof(int));
  GraphHandle nested_fg_cond = BuildNestedCondGraphFalse(res_mgr, i);
  GraphHandle nested_fg_body = BuildNestedBodyGraphFalse(res_mgr, x, i);
  GraphHandle nested_fg_after = BuildNestedAfterGraphFalse(res_mgr, x);
  NodeHandle new_x = MSNewWhile(res_mgr, sub_fg_false, nested_fg_cond, nested_fg_body, nested_fg_after);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_false, new_x, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_false;
}
}

/// Feature: C_API Control Flow
/// Description: test While in If case.
/// Expectation: case works correctly.
TEST_F(TestIfWhile, TestIfWhile) {
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
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 10);
  ASSERT_TRUE(n != nullptr);
  NodeHandle input_nodes[] = {x, n};
  NodeHandle cond = MSNewOp(res_mgr, fg, "Less", input_nodes, 2, NULL, NULL, 0);
  ASSERT_TRUE(cond != nullptr);
  GraphHandle sub_fg_true = BuildTrueGraph(res_mgr, x);
  ASSERT_TRUE(sub_fg_true != nullptr);
  GraphHandle sub_fg_false = BuildFalseGraph(res_mgr, x);
  ASSERT_TRUE(sub_fg_false != nullptr);
  NodeHandle switch_res = MSNewSwitch(res_mgr, fg, cond, sub_fg_true, sub_fg_false);
  ASSERT_TRUE(switch_res != nullptr);

  // NodeHandle z = MSNewScalarConstantInt32(res_mgr, 1);
  // NodeHandle input_nodes_3[] = {switch_res, z};
  // NodeHandle res = MSNewOp(res_mgr, fg, "Add", input_nodes_3, 2, NULL, NULL, 0);
  ret = MSFuncGraphSetOutput(res_mgr, fg, switch_res, false);
  ASSERT_TRUE(ret == RET_OK);

  // test basic funcGraph compiling and executing
  ret = MSFuncGraphCompile(res_mgr, fg);
  ASSERT_TRUE(ret == RET_OK);

  int64_t a[1] = {1};
  int64_t a_shape[1] = {1};
  TensorHandle tensor_a = MSNewTensor(res_mgr, a, MS_INT32, a_shape, 1, 1 * sizeof(int));
  ASSERT_TRUE(tensor_a != nullptr);
  TensorHandle inputs[1] = {tensor_a};
  TensorHandle outputs[1];
  ret = MSFuncGraphRun(res_mgr, fg, inputs, 1, outputs, 1);
  ASSERT_TRUE(ret == RET_OK);
  void *data = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data != nullptr);
  ASSERT_EQ(((int *)data)[0], 1024);
  MSResourceManagerDestroy(res_mgr);
}