/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ut/src/runtime/kernel/opencl/common.h"
#include "nnacl/arg_min_max_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_ArgMinMax : public CommonTest {};

namespace {
// PrimitiveType_ArgMin: src/ops/populate/argmin_populate.cc
// PrimitiveType_ArgMax: src/ops/populate/argmax_populate.cc
OpParameter *CreateParameter(schema::PrimitiveType type, int axis, int topk, bool out_value, bool keep_dims = false,
                             int axis_type = 0) {
  auto *param = test::CreateParameter<ArgMinMaxParameter>(type);
  param->axis_ = axis;
  param->topk_ = topk;
  param->axis_type_ = axis_type;
  param->out_value_ = out_value;
  param->keep_dims_ = keep_dims;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_ArgMinMax, axis0topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 0;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {3, 2, 2, 2};
  std::vector<int> output_shape = {2, 2, 2, 2};
  float input_data[] = {100, 2, 4, 50, 11, 12, 34, 35, 10, 20, 40, 5, 7, 80, 10, 11, 55, 25, 5, 15, 18, 8, 15, 16};
  float output_data[] = {0, 2, 1, 0, 2, 1, 0, 0, 2, 1, 2, 2, 0, 0, 2, 2};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis0topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 0;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {3, 2, 2, 2};
  std::vector<int> output_shape = {2, 2, 2, 2};
  float input_data[] = {100, 2, 4, 50, 11, 12, 34, 35, 10, 20, 40, 5, 7, 80, 10, 11, 55, 25, 5, 15, 18, 8, 15, 16};
  float output_data[] = {100, 25, 40, 50, 18, 80, 34, 35, 55, 20, 5, 15, 11, 12, 15, 16};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis1topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 1;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {2, 3, 2, 3};
  std::vector<int> output_shape = {2, 2, 2, 3};
  float input_data[] = {100, 2,  200, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                        10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  float output_data[] = {0, 1, 0, 1, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 0, 2, 1, 0, 0, 0, 1, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis1topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 1;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {2, 3, 2, 3};
  std::vector<int> output_shape = {2, 2, 2, 3};
  float input_data[] = {100, 2,  200, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                        10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  float output_data[] = {100, 12, 200, 34, 50, 36,  11, 6,  17, 10, 35, 30,
                         18,  80, 90,  40, 22, 120, 10, 20, 30, 10, 11, 60};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 2;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {2, 3, 3, 3};
  std::vector<int> output_shape = {2, 3, 1, 3};
  float input_data[] = {10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12,
                        10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12,
                        10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12};
  float output_data[] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis2topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 2;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 2, 5};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {30, 45, 30, 50, 90, 20, 20, 25, 40, 50, 30, 45, 30, 50, 90, 20, 20, 25, 40, 50,
                         30, 45, 30, 50, 90, 20, 20, 25, 40, 50, 30, 45, 30, 50, 90, 20, 20, 25, 40, 50};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis2topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 2;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 2, 5};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {2, 2, 0, 2, 0, 1, 0, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 0, 1,
                         2, 2, 0, 2, 0, 1, 0, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 0, 1};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis3topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 3;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 3, 2};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis3topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 3;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 3, 2};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {90, 40, 50, 20, 50, 45, 90, 40, 50, 20, 50, 45,
                         90, 40, 50, 20, 50, 45, 90, 40, 50, 20, 50, 45};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim32axis1topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 1;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {1, 2, 14};
  std::vector<int> output_shape = {1, 14};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50,
                        30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25};
  float output_data[] = {1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim43axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 2;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {2, 2, 2, 14};
  std::vector<int> output_shape = {2, 2, 14};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15,
                        1,  50, 30, 45, 25, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30,
                        40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25,
                        50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 10, 20, 30, 40, 90, 20, 11, 15,
                        1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25};
  float output_data[] = {1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
                         1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim21axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 0;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {2, 14};
  std::vector<int> output_shape = {14};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50,
                        30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25};
  float output_data[] = {1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim10axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMax;
  int axis = 0;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {14};
  std::vector<int> output_shape = {1};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50};
  float output_data[] = {4};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
}  // namespace mindspore::lite::opencl::test
