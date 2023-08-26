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

class TestGraphLoad : public ST::Common {
 public:
  TestGraphLoad() {}
};

/// Feature: C_API FuncGraphLoad
/// Description: test TestTensorAdd  case.
/// Expectation: case works correctly.
TEST_F(TestGraphLoad, TestTensorAdd) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();
  
  GraphHandle fg = MSFuncGraphLoad(res_mgr, "/home/workspace/mindspore_dataset/mindir/add/add.mindir");
  ASSERT_TRUE(fg != nullptr);
  float a[4] = {1, 2, 3, 4};
  int64_t a_shape[1] = {4};
  float b[4] = {2, 3, 4, 5};
  int64_t b_shape[1] = {4};
  TensorHandle tensor_a = MSNewTensor(res_mgr, a, MS_FLOAT32, a_shape, 1, 4 * sizeof(float));
  ASSERT_TRUE(tensor_a != nullptr);
  TensorHandle tensor_b = MSNewTensor(res_mgr, b, MS_FLOAT32, b_shape, 1, 4 * sizeof(float));
  ASSERT_TRUE(tensor_b != nullptr);
  TensorHandle inputs[2] = {tensor_a, tensor_b};
  TensorHandle outputs[1];
  ret = MSFuncGraphCompile(res_mgr, fg, NULL, 0);
  ASSERT_TRUE(ret == RET_OK);
  ret = MSFuncGraphRun(res_mgr, fg, inputs, 2, outputs, 1);
  ASSERT_TRUE(ret == RET_OK);
  void *data1 = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data1 != nullptr);
  // result compare
  size_t output_dim = MSTensorGetDimension(res_mgr, outputs[0], &ret);
  int64_t output_shape[output_dim];
  ret = MSTensorGetShape(res_mgr, outputs[0], output_shape, output_dim);
  for (int i=0; i<output_shape[0]; i++){
    ASSERT_EQ(((float *)data1)[i], a[i] + b[i]);
  }
  MSResourceManagerDestroy(res_mgr);
}