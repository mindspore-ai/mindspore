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

class TestSimpleGraph : public ST::Common {
 public:
  TestSimpleGraph() {}
};

/// Feature: C_API Graph
/// Description: test multiple output graph case.
/// Expectation: case works correctly.
TEST_F(TestSimpleGraph, TestMultiOutputs) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();

  // test set context
  ContextAutoSet();

// test basic funcGraph building
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  NodeHandle x = MSNewPlaceholder(res_mgr, fg, MS_INT32, NULL, 0);
  ASSERT_TRUE(x != nullptr);
  NodeHandle y = MSNewScalarConstantInt32(res_mgr, 2);
  ASSERT_TRUE(y != nullptr);
  NodeHandle input_nodes[] = {x, y};
  NodeHandle op1 = MSNewOp(res_mgr, fg, "Add", input_nodes, 2, NULL, NULL, 0);
  ASSERT_TRUE(op1 != nullptr);

  // test makeTuple & tupleGetItem
  NodeHandle tuple = MSPackNodesTuple(res_mgr, fg, input_nodes, 2);
  ASSERT_TRUE(tuple != nullptr);
  NodeHandle item = MSOpGetSpecOutput(res_mgr, fg, tuple, 1);
  ASSERT_TRUE(item != nullptr);

  NodeHandle out_nodes[2] = {op1, item};
  ret = MSFuncGraphSetOutputs(res_mgr, fg, out_nodes, 2, false);
  ASSERT_TRUE(ret == RET_OK);

  // test basic funcGraph compiling and executing
  int64_t a[1] = {97};
  int64_t a_shape[1] = {1};
  ret = MSFuncGraphCompile(res_mgr, fg);
  ASSERT_TRUE(ret == RET_OK);
  TensorHandle tensor_a = MSNewTensor(res_mgr, a, MS_INT32, a_shape, 1, 1 * sizeof(int));
  ASSERT_TRUE(tensor_a != nullptr);
  TensorHandle inputs[1] = {tensor_a};
  TensorHandle outputs[2];
  ret = MSFuncGraphRun(res_mgr, fg, inputs, 1, outputs, 2);
  ASSERT_TRUE(ret == RET_OK);
  void *data1 = MSTensorGetData(res_mgr, outputs[0]);
  ASSERT_TRUE(data1 != nullptr);
  ASSERT_EQ(((int *)data1)[0], 99);
  void *data2 = MSTensorGetData(res_mgr, outputs[1]);
  ASSERT_TRUE(data2 != nullptr);
  ASSERT_EQ(((int *)data2)[0], 2);
  MSResourceManagerDestroy(res_mgr);
}

/// Feature: C_API Graph
/// Description: test convolution graph case.
/// Expectation: case works correctly.
TEST_F(TestSimpleGraph, TestConvReLU) {
  STATUS ret;
  ResMgrHandle res_mgr = MSResourceManagerCreate();
  ASSERT_TRUE(res_mgr != nullptr);

  // test set context
  ContextAutoSet();

  // === Testing Conv2D ===
  GraphHandle fg = MSFuncGraphCreate(res_mgr);
  ASSERT_TRUE(fg != nullptr);
  // conv variable settings
  int64_t x_shape[] = {1, 1, 3, 3};
  int64_t w_shape[] = {1, 1, 2, 2};
  float w_data[] = {1, 1, 1, 1};
  NodeHandle x_conv = MSNewPlaceholder(res_mgr, fg, MS_FLOAT32, x_shape, 4);
  ASSERT_TRUE(x_conv != nullptr);
  NodeHandle w = MSNewTensorVariable(res_mgr, fg, w_data, MS_FLOAT32, w_shape, 4, 4 * sizeof(float));
  ASSERT_TRUE(w != nullptr);

  // Conv Attribute Settings
  AttrHandle out_channel = MSNewAttrInt64(res_mgr, 1);
  ASSERT_TRUE(out_channel != nullptr);
  int64_t kernel_size_raw[] = {2, 2};
  AttrHandle kernel_size = MSNewAttrArray(res_mgr, kernel_size_raw, 2, MS_INT64);
  ASSERT_TRUE(kernel_size != nullptr);
  AttrHandle mode = MSNewAttrInt64(res_mgr, 1);
  ASSERT_TRUE(mode != nullptr);
  AttrHandle pad_mode = MSNewAttrInt64(res_mgr, VALID);
  ASSERT_TRUE(pad_mode != nullptr);
  int64_t pad_raw[] = {0, 0, 0, 0};
  AttrHandle pad = MSNewAttrArray(res_mgr, pad_raw, 4, MS_INT64);
  ASSERT_TRUE(pad != nullptr);
  int64_t stride_raw[] = {1, 1, 1, 1};
  AttrHandle stride = MSNewAttrArray(res_mgr, stride_raw, 4, MS_INT64);
  ASSERT_TRUE(stride != nullptr);
  int64_t dilation_raw[] = {1, 1, 1, 1};
  AttrHandle dilation = MSNewAttrArray(res_mgr, dilation_raw, 4, MS_INT64);
  ASSERT_TRUE(dilation != nullptr);
  AttrHandle group = MSNewAttrInt64(res_mgr, 1);
  ASSERT_TRUE(group != nullptr);
  AttrHandle format = MSNewAttrInt64(res_mgr, NCHW);
  ASSERT_TRUE(format != nullptr);
  const char *attr_names[] = {"out_channel", "kernel_size", "mode",  "pad_mode", "pad",
                              "stride",      "dilation",    "group", "format"};
  AttrHandle attrs[] = {out_channel, kernel_size, mode, pad_mode, pad, stride, dilation, group, format};
  size_t attr_num = 9;

  NodeHandle conv_input_nodes[] = {x_conv, w};
  size_t conv_input_num = 2;
  NodeHandle op_conv = MSNewOp(res_mgr, fg, "Conv2D", conv_input_nodes, conv_input_num, attr_names, attrs, attr_num);
  ASSERT_TRUE(format != op_conv);

  // relu variable settings
  NodeHandle relu_input_nodes[] = {op_conv};
  NodeHandle op_relu = MSNewOp(res_mgr, fg, "ReLU", relu_input_nodes, 1, NULL, NULL, 0);
  ASSERT_TRUE(format != op_relu);

  // Graph output
  ret = MSFuncGraphSetOutput(res_mgr, fg, op_relu, false);
  ASSERT_TRUE(ret == RET_OK);

  ret = MSFuncGraphCompile(res_mgr, fg);
  ASSERT_TRUE(ret == RET_OK);

  // Graph input data
  float data_value[] = {1, 1, 1, 1, 1, -1, -1, -1, -1};
  int64_t data_shape[] = {1, 1, 3, 3};
  TensorHandle input_tensor = MSNewTensor(res_mgr, data_value, MS_FLOAT32, data_shape, 4, 9 * sizeof(float));
  ASSERT_TRUE(input_tensor != nullptr);
  TensorHandle graph_input[] = {input_tensor};
  TensorHandle graph_output[1];
  ret = MSFuncGraphRun(res_mgr, fg, graph_input, 1, graph_output, 1);
  ASSERT_TRUE(ret == RET_OK);

  size_t dim = MSTensorGetDimension(res_mgr, graph_output[0], &ret);
  ASSERT_EQ(dim, 4);
  int64_t output_shape[4];
  ret = MSTensorGetShape(res_mgr, graph_output[0], output_shape, 4);
  ASSERT_EQ(output_shape[0], 1);
  ASSERT_EQ(output_shape[1], 1);
  ASSERT_EQ(output_shape[2], 2);
  ASSERT_EQ(output_shape[3], 2);

  void *result = MSTensorGetData(res_mgr, graph_output[0]);
  ASSERT_TRUE(result != nullptr);
  ASSERT_EQ(((float *)result)[0], 4);
  ASSERT_EQ(((float *)result)[1], 2);
  ASSERT_EQ(((float *)result)[2], 0);
  ASSERT_EQ(((float *)result)[3], 0);
  MSResourceManagerDestroy(res_mgr);
}
