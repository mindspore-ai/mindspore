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
#include "nnacl/softmax_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_SoftMax : public CommonTest {};

namespace {
// PrimitiveType_SoftMax: src/ops/populate/softmax_populate.cc
OpParameter *CreateParameter(int axis) {
  auto *param = test::CreateParameter<SoftmaxParameter>(schema::PrimitiveType_Softmax);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_SoftMax, 2D_axis1) {
  int axis = 1;
  std::vector<int> input_shape = {1, 10};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float output_data[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 2e-2 : 1e-5);
  }
}

TEST_F(TestOpenCL_SoftMax, 4D_axis3) {
  int axis = 3;
  std::vector<int> input_shape = {1, 2, 1, 5};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float output_data[] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 2e-2 : 1e-5);
  }
}

TEST_F(TestOpenCL_SoftMax, 4D_axis1) {
  int axis = 1;
  std::vector<int> input_shape = {1, 2, 1, 1};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 1};
  float output_data[] = {0.5, 0.5};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 2e-2 : 1e-5);
  }
}

TEST_F(TestOpenCL_SoftMax, 2D_axis1_N) {
  int axis = 1;
  std::vector<int> input_shape = {2, 10};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float output_data[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 2e-2 : 1e-5);
  }
}

TEST_F(TestOpenCL_SoftMax, 4D_axis3_N) {
  int axis = 3;
  std::vector<int> input_shape = {2, 2, 1, 5};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float output_data[] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                         0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 2e-2 : 1e-5);
  }
}

TEST_F(TestOpenCL_SoftMax, 4D_axis1_N) {
  int axis = 1;
  std::vector<int> input_shape = {2, 2, 1, 1};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 1, 1, 1};
  float output_data[] = {0.5, 0.5, 0.5, 0.5};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 2e-2 : 1e-5);
  }
}
}  // namespace mindspore::lite::opencl::test
