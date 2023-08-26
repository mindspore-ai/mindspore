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

class TestCustomOp : public ST::Common {
 public:
  TestCustomOp() {}
};

namespace {
STATUS CustomAddInferType(const DataTypeC *input_types, size_t input_num, DataTypeC *output_types, size_t output_num) {
  if (input_num != 2 || output_num != 1) {
    return RET_ERROR;
  }
  output_types[0] = input_types[0];
  return RET_OK;
}

STATUS CustomAddInferShape(int64_t **input_shapes, const size_t *input_dims, size_t input_num, int64_t **output_shapes,
                           size_t *output_dims, size_t output_num) {
  if (input_num != 2 || output_num != 1 || input_dims == NULL) {
    return RET_ERROR;
  }
  output_dims[0] = input_dims[0];
  for (int i = 0; i < output_dims[0]; i++) {
    output_shapes[0][i] = input_shapes[0][i];
  }
  return RET_OK;
}
}

/// Feature: C_API Graph
/// Description: test cpu aot custom op case.
/// Expectation: case works correctly.
TEST_F(TestCustomOp, TestCPUAotCustomOp) {
  // pass case if device target is Ascend
  std::string device_target = std::getenv("DEVICE_TARGET");
  if (device_target == "Ascend" || device_target == "ascend") {
    return;
  }

  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();

  // setting custom info
  CustomOpInfo info;
  const char *input_name[] = {"x1", "x2"};
  info.input_names = input_name;
  info.input_num = 2;
  const char *output_name[] = {"y"};
  info.output_names = output_name;
  info.output_num = 1;
  DTypeFormat dtype_format_1[] = {None_None, None_None, None_None};
  DTypeFormat dtype_format_2[] = {F32_None, F32_None, F32_None};
  DTypeFormat *dtype_format[] = {dtype_format_1, dtype_format_2};
  info.dtype_formats = dtype_format;
  info.dtype_formats_num = 2;
  const char *attr_name[] = {"scale", "paddings"};
  info.attr_names = attr_name;
  ValueHandle scale = MSNewValueFloat32(res_mgr, 0.7f);
  float pad[] = {2, 2};
  ValueHandle paddings = MSNewValueArray(res_mgr, pad, 2, MS_FLOAT32);
  ValueHandle attrs[] = {scale, paddings};
  info.attr_values = attrs;
  info.attr_num = 2;
  info.target = "CPU";
  info.func_type = "aot";
  info.func_name = "./libcustom_add.so:CustomAdd";
  info.dtype_infer_func = CustomAddInferType;
  info.shape_infer_func = CustomAddInferShape;
  info.output_shapes = NULL;
  info.output_dtypes = NULL;

  // building graph
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  NodeHandle x = MSNewPlaceholder(res_mgr, fg, MS_FLOAT32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle y = MSNewConstantScalarFloat32(res_mgr, 10);
  ASSERT_TRUE(y != nullptr);
  NodeHandle input_nodes[] = {x, y};
  NodeHandle op = MSNewCustomOp(res_mgr, fg, input_nodes, 2, info);
  ASSERT_TRUE(op != nullptr);

  ret = MSFuncGraphSetOutput(res_mgr, fg, op, false);
  ASSERT_TRUE(ret == RET_OK);

  // test basic funcGraph compiling and executing
  float a[1] = {2};
  int64_t a_shape[1] = {1};
  ret = MSFuncGraphCompile(res_mgr, fg, NULL, 0);
  ASSERT_TRUE(ret == RET_OK);
  TensorHandle tensor_a = MSNewTensor(res_mgr, a, MS_FLOAT32, a_shape, 1, 1 * sizeof(float));
  ASSERT_TRUE(tensor_a != nullptr);
  TensorHandle inputs[1] = {tensor_a};
  TensorHandle outputs[1];
  ret = MSFuncGraphRun(res_mgr, fg, inputs, 1, outputs, 1);
  ASSERT_TRUE(ret == RET_OK);
  void *data = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data != nullptr);
  ASSERT_EQ(((float *)data)[0], 20);
  MSResourceManagerDestroy(res_mgr);
}
