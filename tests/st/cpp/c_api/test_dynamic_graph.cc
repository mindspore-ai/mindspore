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
#include "common/utils.h"
#include "include/c_api/ms/graph.h"
#include "include/c_api/ms/node.h"
#include "include/c_api/ms/tensor.h"
#include "include/c_api/ms/context.h"
#include "include/c_api/ms/base/status.h"
#include "include/c_api/ms/base/handle_types.h"
#include "include/c_api/ms/value.h"

class TestDynamicGraph : public ST::Common {
 public:
  TestDynamicGraph() {}
};

/// Feature: C_API Dynamic Graph
/// Description: test simple dynamic op.
/// Expectation: case works correctly.
TEST_F(TestDynamicGraph, TestSimpleDynamicOp) {
STATUS ret;
ResMgrHandle res_mgr = MSResourceManagerCreate();
ASSERT_TRUE(res_mgr != nullptr);

// test set context
ContextAutoSet();

// prepare input
float value_a[] = {1, 2};
int64_t shape_a[] = {1, 2};
TensorHandle a = MSNewTensor(res_mgr, value_a, MS_FLOAT32, shape_a, 2, 2 * sizeof(float));
ASSERT_TRUE(a != nullptr);
float value_b[] = {2, 3};
int64_t shape_b[] = {1, 2};
TensorHandle b = MSNewTensor(res_mgr, value_b, MS_FLOAT32, shape_b, 2, 2 * sizeof(float));
ASSERT_TRUE(b != nullptr);
TensorHandle inputs[2] = {a, b};
TensorHandle outputs[1];

// call op
ret = MSRunOp(res_mgr, "Add", inputs, 2, outputs, 1);
ASSERT_TRUE(ret == RET_OK);
void *data = MSTensorGetData(res_mgr, outputs[0]);
ASSERT_TRUE(data != nullptr);
ASSERT_EQ(((float *)data)[0], 3);
int64_t output_shape[2];
ret = MSTensorGetShape(res_mgr, outputs[0], output_shape, 2);
ASSERT_TRUE(ret == RET_OK);
ASSERT_EQ(output_shape[0], 1);
ASSERT_EQ(output_shape[1], 2);

MSResourceManagerDestroy(res_mgr);
}

/// Feature: C_API Dynamic Graph
/// Description: test simple dynamic op and cache mechanism.
/// Expectation: case works correctly.
TEST_F(TestDynamicGraph, TestSimpleDynamicOpWithCache) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();

  // configure info
  DynamicOpInfo info;
  const char *attr_name[] = {"transpose_a", "transpose_b"};
  info.attr_names = attr_name;
  ValueHandle transpose_a = MSNewValueBool(res_mgr, false);
  ValueHandle transpose_b = MSNewValueBool(res_mgr, true);
  ValueHandle attrs[] = {transpose_a, transpose_b};
  info.attr_values = attrs;
  info.attr_num = 2;
  info.output_shapes = NULL;
  info.output_dims = NULL;
  info.output_dtypes = NULL;

  // prepare input
  float value_a[] = {1, 2};
  int64_t shape_a[] = {1, 2};
  TensorHandle a = MSNewTensor(res_mgr, value_a, MS_FLOAT32, shape_a, 2, 2 * sizeof(float));
  ASSERT_TRUE(a != nullptr);
  float value_b[] = {2, 3};
  int64_t shape_b[] = {1, 2};
  TensorHandle b = MSNewTensor(res_mgr, value_b, MS_FLOAT32, shape_b, 2, 2 * sizeof(float));
  ASSERT_TRUE(b != nullptr);
  TensorHandle inputs_1[2] = {a, b};
  TensorHandle outputs_1[1];

  // call op
  ret = MSRunOpWithInfo(res_mgr, "MatMul", inputs_1, 2, outputs_1, 1, info);
  ASSERT_TRUE(ret == RET_OK);
  void *data_1 = MSTensorGetData(res_mgr, outputs_1[0]);
  ASSERT_TRUE(data_1 != nullptr);
  ASSERT_EQ(((float *)data_1)[0], 8);
  int64_t output_shape[2];
  ret = MSTensorGetShape(res_mgr, outputs_1[0], output_shape, 2);
  ASSERT_TRUE(ret == RET_OK);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 1);

  // test op cache function
  size_t num_before = GetCachedOpNum(res_mgr);
  ASSERT_EQ(num_before, 1);

  // change op info
  int64_t out_shape[] = {2, 2};
  int64_t *output_shapes[] = {out_shape};
  info.output_shapes = output_shapes;
  size_t output_dims[] = {2};
  info.output_dims = output_dims;
  DataTypeC out_type[] = {MS_FLOAT32};
  info.output_dtypes = out_type;

  // change op inputs
  float value_c[] = {1, 2, 3, 4, 5, 6};
  int64_t shape_c[] = {2, 3};
  TensorHandle c = MSNewTensor(res_mgr, value_c, MS_FLOAT32, shape_c, 2, 6 * sizeof(float));
  ASSERT_TRUE(c != nullptr);
  float value_d[] = {4, 5, 6, 7, 8, 9};
  int64_t shape_d[] = {2, 3};
  TensorHandle d = MSNewTensor(res_mgr, value_d, MS_FLOAT32, shape_d, 2, 6 * sizeof(float));
  ASSERT_TRUE(d != nullptr);
  TensorHandle inputs_2[2] = {c, d};
  TensorHandle outputs_2[1];

  for (size_t i = 0; i < 10; i++) {
    ret = MSRunOpWithInfo(res_mgr, "MatMul", inputs_2, 2, outputs_2, 1, info);
    ASSERT_TRUE(ret == RET_OK);
    void *data_2 = MSTensorGetData(res_mgr, outputs_2[0]);
    ASSERT_TRUE(data_2 != nullptr);
    ASSERT_EQ(((float *)data_2)[0], 32);
    ASSERT_EQ(((float *)data_2)[3], 122);
  }

  size_t num_after = GetCachedOpNum(res_mgr);
  ASSERT_EQ(num_after, 2);

  MSResourceManagerDestroy(res_mgr);
}

/// Feature: C_API Dynamic Graph
/// Description: test const input to attribute.
/// Expectation: case works correctly.
TEST_F(TestDynamicGraph, TestConstInputToAttr) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();

  // configure info
  DynamicOpInfo info;
  const char *attr_name[] = {"input_names"};
  info.attr_names = attr_name;
  const char *in_names[] = {"x", "dst_type"};
  ValueHandle input_names = MSNewValueStrings(res_mgr, in_names, 2);
  ValueHandle attrs[] = {input_names};
  info.attr_values = attrs;
  info.attr_num = 1;
  int64_t out_shape[] = {};
  int64_t *output_shapes[] = {out_shape};
  info.output_shapes = output_shapes;
  size_t output_dims[] = {0};
  info.output_dims = output_dims;
  DataTypeC out_type[] = {MS_FLOAT32};
  info.output_dtypes = out_type;

  // prepare input
  TensorHandle a = MSNewTensorScalarInt32(res_mgr, 6);
  ASSERT_TRUE(a != nullptr);
  TensorHandle b = MSNewValueType(res_mgr, MS_INT32);
  ASSERT_TRUE(b != nullptr);
  TensorHandle inputs[2] = {a, b};
  TensorHandle outputs[1];

  // call op
  ret = MSRunOpWithInfo(res_mgr, "Cast", inputs, 2, outputs, 1, info);
  ASSERT_TRUE(ret == RET_OK);
  void *data = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data != nullptr);
  ASSERT_EQ(((float *)data)[0], 6);
  DataTypeC type = MSTensorGetDataType(res_mgr, outputs[0], &ret);
  ASSERT_TRUE(ret == RET_OK);
  ASSERT_EQ(type, MS_FLOAT32);

  MSResourceManagerDestroy(res_mgr);
}
