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

class TestWhile : public ST::Common {
 public:
  TestWhile() {}
};

/*
 * the pseudo-code to be implemented
 * m = 1
 * n = 3
 * extra = m + n
 * x = 97
 * i = 1
 * while (i < 4):
 *   x += extra
 *   i += 1
 * return x
 */
namespace {
GraphHandle BuildCondGraph(ResMgrHandle res_mgr, NodeHandle i) {
  GraphHandle sub_fg_cond = MSFuncGraphCreate(res_mgr);
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 4);
  NodeHandle input_nodes[] = {i, n};
  NodeHandle cond = MSNewOp(res_mgr, sub_fg_cond, "Less", input_nodes, 2, NULL, NULL, 0);
  STATUS ret = MSFuncGraphSetOutput(res_mgr, sub_fg_cond, cond, false);
  if (ret != RET_OK) {
    MSResourceManagerDestroy(res_mgr);
    return NULL;
  }
  return sub_fg_cond;
}

GraphHandle BuildBodyGraph(ResMgrHandle res_mgr, NodeHandle extra, NodeHandle x, NodeHandle i) {
  GraphHandle sub_fg_body = MSFuncGraphCreate(res_mgr);
  NodeHandle step = MSNewScalarConstantInt32(res_mgr, 1);
  NodeHandle input_nodes_body[] = {x, extra};
  NodeHandle new_x = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_body, 2, NULL, NULL, 0);
  NodeHandle input_nodes_i[] = {i, step};
  NodeHandle new_i = MSNewOp(res_mgr, sub_fg_body, "Add", input_nodes_i, 2, NULL, NULL, 0);
  NodeHandle out_nodes[3] = {new_x, extra, new_i};
  STATUS ret = MSFuncGraphSetOutputs(res_mgr, sub_fg_body, out_nodes, 3, false);
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
/// Description: test While case.
/// Expectation: case works correctly.
TEST_F(TestWhile, TestWhile) {
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
  NodeHandle m = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(m != nullptr);
  NodeHandle n = MSNewScalarConstantInt32(res_mgr, 3);
  ASSERT_TRUE(n != nullptr);
  NodeHandle input_nodes_extra[] = {m, n};
  NodeHandle extra_var = MSNewOp(res_mgr, fg, "Add", input_nodes_extra, 2, NULL, NULL, 0);
  ASSERT_TRUE(extra_var != nullptr);
  // cond branch
  GraphHandle sub_fg_cond = BuildCondGraph(res_mgr, i);
  ASSERT_TRUE(sub_fg_cond != nullptr);
  // as shown below, you can directly define the condition part in main graph as well.
  // NodeHandle n = MSNewScalarConstantInt32(res_mgr, 4);
  // NodeHandle input_nodes_1[] = {i, n};
  // NodeHandle cond = MSNewOp(res_mgr, fg, "Less", input_nodes_1, 2, NULL, NULL, 0);
  // body branch
  GraphHandle sub_fg_body = BuildBodyGraph(res_mgr, extra_var, x, i);
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

  int64_t a[1] = {97};
  int64_t b[1] = {1};
  int64_t shape[1] = {1};
  TensorHandle tensor_a = MSNewTensor(res_mgr, a, MS_INT32, shape, 1, 1 * sizeof(int));
  ASSERT_TRUE(tensor_a != nullptr);
  TensorHandle tensor_b = MSNewTensor(res_mgr, b, MS_INT32, shape, 1, 1 * sizeof(int));
  ASSERT_TRUE(tensor_b != nullptr);
  TensorHandle inputs[] = {tensor_a, tensor_b};
  TensorHandle outputs[1];
  ret = MSFuncGraphRun(res_mgr, fg, inputs, 2, outputs, 1);
  ASSERT_TRUE(ret == RET_OK);
  void *data = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data != nullptr);
  ASSERT_EQ(((int *)data)[0], 113);
  MSResourceManagerDestroy(res_mgr);
}