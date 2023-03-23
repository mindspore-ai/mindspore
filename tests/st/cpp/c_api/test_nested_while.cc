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

class TestNestedWhile : public ST::Common {
 public:
  TestNestedWhile() {}
};

/*
 * the pseudo-code to be implemented
 * x = 1
 * for (int i=0; i<8; ++i) {
 *   for (int j=0; j<3; ++j)
 *     x += 2
 *   ++x
 *  }
 * return x
 */
namespace {
GraphHandle BuildNestedCondGraph(ResMgrHandle res_mgr, NodeHandle j) {
  GraphHandle sub_fg_cond = MSFuncGraphCreate(res_mgr);
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 3);
  NodeHandle input_nodes[] = {j, n};
  NodeHandle cond = MSNewOp(res_mgr, sub_fg_cond, "Less", input_nodes, 2, NULL, NULL, 0);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_cond, cond, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_cond;
}

GraphHandle BuildNestedBodyGraph(ResMgrHandle res_mgr, NodeHandle x, NodeHandle j) {
  GraphHandle sub_fg_body = MSFuncGraphCreate(res_mgr);
  NodeHandle step1 = MSNewScalarConstantInt32(res_mgr, 2);
  NodeHandle step2 = MSNewScalarConstantInt32(res_mgr, 1);
  NodeHandle input_nodes_body[] = {x, step1};
  NodeHandle new_x = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_body, 2, NULL, NULL, 0);
  NodeHandle input_nodes_j[] = {j, step2};
  NodeHandle new_j = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_j, 2, NULL, NULL, 0);
  NodeHandle out_nodes[2] = {new_x, new_j};
  STATUS ret = MSFuncGraphSetOutputs(res_mgr, sub_fg_body, out_nodes, 2, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_body;
}

GraphHandle BuildNestedAfterGraph(ResMgrHandle res_mgr, NodeHandle x) {
  GraphHandle sub_fg_after = MSFuncGraphCreate(res_mgr);
  NodeHandle step = MSNewScalarConstantInt32(res_mgr, 1);
  NodeHandle input_nodes_after[] = {x, step};
  NodeHandle new_x = MSNewOp(res_mgr, sub_fg_after, "Add", input_nodes_after, 2, NULL, NULL, 0);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_after, new_x, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_after;
}

GraphHandle BuildCondGraph(ResMgrHandle res_mgr, NodeHandle i) {
  GraphHandle sub_fg_cond = MSFuncGraphCreate(res_mgr);
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 8);
  NodeHandle input_nodes[] = {i, n};
  NodeHandle cond = MSNewOp(res_mgr, sub_fg_cond, "Less", input_nodes, 2, NULL, NULL, 0);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_cond, cond, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_cond;
}

GraphHandle BuildBodyGraph(ResMgrHandle res_mgr, NodeHandle x, NodeHandle i) {
  GraphHandle sub_fg_body = MSFuncGraphCreate(res_mgr);
  NodeHandle step = MSNewScalarConstantInt32(res_mgr, 1);
  int data_j[] = {0};
  int64_t shape_j[] = {1};
  NodeHandle j = MSNewTensorVariable(res_mgr, sub_fg_body, data_j, MS_INT32, shape_j, 1, 1 * sizeof(int));
  GraphHandle nested_fg_cond = BuildNestedCondGraph(res_mgr, j);
  GraphHandle nested_fg_body = BuildNestedBodyGraph(res_mgr, x, j);
  GraphHandle nested_fg_after = BuildNestedAfterGraph(res_mgr, x);
  NodeHandle new_x = MSNewWhile(res_mgr, sub_fg_body, nested_fg_cond, nested_fg_body, nested_fg_after);
  NodeHandle input_nodes_i[] = {i, step};
  NodeHandle new_i = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_i, 2, NULL, NULL, 0);
  NodeHandle out_nodes[2] = {new_x, new_i};
  STATUS ret = MSFuncGraphSetOutputs(res_mgr, sub_fg_body, out_nodes, 2, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_body;
}

GraphHandle BuildAfterGraph(ResMgrHandle res_mgr, NodeHandle x) {
  GraphHandle sub_fg_after = MSFuncGraphCreate(res_mgr);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_after, x, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_after;
}
}

/// Feature: C_API Control Flow
/// Description: test nested While case.
/// Expectation: case works correctly.
TEST_F(TestNestedWhile, TestNestedWhile) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();

  // main graph
  int data_i[] = {0};
  int64_t shape_i[] = {1};
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  NodeHandle x = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle i = MSNewTensorVariable(res_mgr, fg, data_i, MS_INT32, shape_i, 1, 1 * sizeof(int));
  ASSERT_TRUE(i != nullptr);
  // cond branch
  GraphHandle sub_fg_cond = BuildCondGraph(res_mgr, i);
  ASSERT_TRUE(sub_fg_cond != nullptr);
  // body branch
  GraphHandle sub_fg_body = BuildBodyGraph(res_mgr, x, i);
  ASSERT_TRUE(sub_fg_body != nullptr);
  // after branch
  GraphHandle sub_fg_after = BuildAfterGraph(res_mgr, x);
  ASSERT_TRUE(sub_fg_after != nullptr);
  // call while
  NodeHandle while_res = MSNewWhile(res_mgr, fg, sub_fg_cond, sub_fg_body, sub_fg_after);
  ASSERT_TRUE(while_res != nullptr);
  ret = MSFuncGraphSetOutput(res_mgr, fg, while_res, false);
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
  ASSERT_EQ(((int *)data)[0], 57);
  MSResourceManagerDestroy(res_mgr);
}