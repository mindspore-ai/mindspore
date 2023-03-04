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
#include "c_api/include/node.h"
#include "c_api/include/tensor.h"
#include "c_api/include/context.h"
#include "c_api/base/status.h"
#include "c_api/base/handle_types.h"

namespace mindspore {
class TestCApiNode : public UT::Common {
 public:
  TestCApiNode() = default;
};

/// Feature: C_API
/// Description: test op create, set and get method.
/// Expectation: op create/set/get works correctly.
TEST_F(TestCApiNode, test_op_node) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  NodeHandle x = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle y = MSNewScalarConstantInt32(res_mgr, 3);
  ASSERT_TRUE(y != nullptr);
  NodeHandle input_nodes[] = {x, y};
  // test normal operator
  NodeHandle op = MSNewOp(res_mgr, fg, "Add", input_nodes, 2, NULL, NULL, 0);
  ASSERT_TRUE(op != nullptr);
  // test op get inputs
  NodeHandle input_node1 = MSOpGetInput(res_mgr, op, 1);
  ASSERT_TRUE(input_node1 == y);
  int value = MSScalarConstantGetValueInt32(res_mgr, input_node1, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(value, 3);
  size_t input_num = MSOpGetInputsNum(res_mgr, op, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_TRUE(input_num == 2);
  NodeHandle inputs[2];
  ret = MSOpGetInputs(res_mgr, op, inputs, input_num);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_TRUE(inputs[0] == x);
  ASSERT_TRUE(inputs[1] == y);
  // test op set/get name
  ret = MSOpSetName(res_mgr, op, "hehe");
  ASSERT_EQ(ret, RET_OK);
  char str_buf[5];
  ret = MSNodeGetName(res_mgr, op, str_buf, 5);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(std::string(str_buf), "hehe");
  MSResourceManagerDestroy(res_mgr);
}

/// Feature: C_API
/// Description: test nodes create, set and get method.
/// Expectation: create/set/get works correctly.
TEST_F(TestCApiNode, test_normal_nodes) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  // test Tensor Variable
  int64_t a_shape[] = {1, 2};
  float a_data[] = {1.2, 3.4};
  NodeHandle a1 = MSNewTensorVariable(res_mgr, fg, a_data, MS_FLOAT32, a_shape, 2, 2 * sizeof(float));
  ASSERT_TRUE(a1 != nullptr);
  TensorHandle tensor1 = MSNewTensor(res_mgr, a_data, MS_FLOAT32, a_shape, 2, 2 * sizeof(float));
  ASSERT_TRUE(tensor1 != nullptr);
  NodeHandle a2 = MSNewTensorVariableFromTensor(res_mgr, fg, tensor1);
  ASSERT_TRUE(a2 != nullptr);
  size_t a_size = MSTensorVariableGetDataSize(res_mgr, a1, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(a_size, 2 * sizeof(float));
  void *a1_data = MSTensorVariableGetData(res_mgr, a1);
  ASSERT_TRUE(a1_data != nullptr);
  void *a2_data = MSTensorVariableGetData(res_mgr, a2);
  ASSERT_TRUE(a2_data != nullptr);
  float *a1_data_f = static_cast<float *>(a1_data);
  float *a2_data_f = static_cast<float *>(a2_data);
  ASSERT_EQ(a1_data_f[0], 1.2f);
  ASSERT_EQ(a1_data_f[1], 3.4f);
  ASSERT_EQ(a2_data_f[0], 1.2f);
  ASSERT_EQ(a2_data_f[1], 3.4f);
  // test Tensor Constant
  int64_t b_shape[] = {1, 2};
  int b_data[] = {4, 3};
  NodeHandle b1 = MSNewTensorConstant(res_mgr, b_data, MS_INT32, b_shape, 2, 2 * sizeof(int));
  ASSERT_TRUE(b1 != nullptr);
  TensorHandle tensor2 = MSNewTensor(res_mgr, b_data, MS_INT32, b_shape, 2, 2 * sizeof(int));
  ASSERT_TRUE(tensor2 != nullptr);
  NodeHandle b2 = MSNewTensorConstantFromTensor(res_mgr, tensor2);
  ASSERT_TRUE(b2 != nullptr);
  size_t b_size = MSTensorConstantGetDataSize(res_mgr, b1, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(b_size, 2 * sizeof(int));
  void *b1_data = MSTensorConstantGetData(res_mgr, b1);
  ASSERT_TRUE(b1_data != nullptr);
  void *b2_data = MSTensorConstantGetData(res_mgr, b2);
  ASSERT_TRUE(b2_data != nullptr);
  int *b1_data_f = static_cast<int *>(b1_data);
  int *b2_data_f = static_cast<int *>(b2_data);
  ASSERT_EQ(b1_data_f[0], 4);
  ASSERT_EQ(b1_data_f[1], 3);
  ASSERT_EQ(b2_data_f[0], 4);
  ASSERT_EQ(b2_data_f[1], 3);
  // test other Constants
  NodeHandle x1 = MSNewScalarConstantInt64(res_mgr, 3);
  ASSERT_TRUE(x1 != nullptr);
  int64_t value1 = MSScalarConstantGetValueInt64(res_mgr, x1, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(value1, 3);
  NodeHandle x2 = MSNewScalarConstantFloat32(res_mgr, 3);
  ASSERT_TRUE(x2 != nullptr);
  float value2 = MSScalarConstantGetValueFloat32(res_mgr, x2, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(value2, 3);
  NodeHandle x3 = MSNewScalarConstantBool(res_mgr, true);
  ASSERT_TRUE(x3 != nullptr);
  bool value3 = MSScalarConstantGetValueBool(res_mgr, x3, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(value3, true);
  NodeHandle x4 = MSNewStringConstant(res_mgr, "haha");
  ASSERT_TRUE(x4 != nullptr);
  char value_4[5];
  ret = MSStringConstantGetValue(res_mgr, x4, value_4, 5);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(std::string(value_4), "haha");
  int64_t vec[] = {6, 7};
  NodeHandle x5 = MSNewTupleConstantInt64(res_mgr, vec, 2);
  ASSERT_TRUE(x5 != nullptr);
  size_t tuple_size = MSTupleConstantGetSize(res_mgr, x5, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(tuple_size, 2);
  int64_t vec_get[2];
  ret = MSTupleConstantGetValueInt64(res_mgr, x5, vec_get, 2);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(vec_get[0], vec[0]);
  ASSERT_EQ(vec_get[1], vec[1]);
  NodeHandle x6 = MSNewTypeConstant(res_mgr, MS_INT32);
  ASSERT_TRUE(x6 != nullptr);
  DataTypeC value_6 = MSTypeConstantGetValue(res_mgr, x6, &ret);
  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(value_6, MS_INT32);
  MSResourceManagerDestroy(res_mgr);
}
}
