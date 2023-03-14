/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <cmath>
#include <memory>
#include <sstream>
#include <unordered_map>
#include "common/common_test.h"
#include "c_api/include/tensor.h"
#include "c_api/include/graph.h"
#include "c_api/include/context.h"
#include "c_api/base/status.h"
#include "c_api/base/handle_types.h"

namespace mindspore {
class TestCApiGraph : public UT::CApiCommon {
 public:
  TestCApiGraph() = default;
};

/// Feature: C_API
/// Description: test graph create, get inputs, set/get outputs method.
/// Expectation: graph create, get inputs, set/get outputs works correctly.
TEST_F(TestCApiGraph, test_multi_output_graph) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  NodeHandle x = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle y = MSNewScalarConstantInt32(res_mgr, 2);
  ASSERT_TRUE(y != nullptr);
  NodeHandle input_nodes[] = {x, y};
  // test normal operator
  NodeHandle op1 = MSNewOp(res_mgr, fg, "Add", input_nodes, 2, NULL, NULL, 0);
  ASSERT_TRUE(op1 != nullptr);
  // test makeTuple & tupleGetItem
  NodeHandle tuple = MSPackNodesTuple(res_mgr, fg, input_nodes, 2);
  ASSERT_TRUE(tuple != nullptr);
  NodeHandle item = MSOpGetSpecOutput(res_mgr, fg, tuple, 1);
  ASSERT_TRUE(item != nullptr);
  // test set multi-output
  NodeHandle output_nodes[2] = {op1, item};
  ret = MSFuncGraphSetOutputs(res_mgr, fg, output_nodes, 2, false);
  ASSERT_EQ(ret, RET_OK);
  // test get input
  NodeHandle input_0 = MSFuncGraphGetInput(res_mgr, fg, 0);
  ASSERT_TRUE(input_0 == x);
  size_t input_num = MSFuncGraphGetInputNum(res_mgr, fg, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(input_num, 1);
  NodeHandle in_nodes[1];
  ret = MSFuncGraphGetInputs(res_mgr, fg, in_nodes, 1);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_TRUE(in_nodes[0] == x);
  // test get output
  NodeHandle output_1 = MSFuncGraphGetOutput(res_mgr, fg, 0);
  ASSERT_TRUE(output_1 == op1);
  NodeHandle output_2 = MSFuncGraphGetOutput(res_mgr, fg, 1);
  ASSERT_TRUE(output_2 == item);
  size_t output_num = MSFuncGraphGetOutputNum(res_mgr, fg, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(output_num, 2);
  NodeHandle out_nodes[2];
  ret = MSFuncGraphGetOutputs(res_mgr, fg, out_nodes, 2);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_TRUE(out_nodes[0] == op1);
  ASSERT_TRUE(out_nodes[1] == item);
  MSResourceManagerDestroy(res_mgr);
}
}
