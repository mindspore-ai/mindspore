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
#include "nnacl/stack_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Stack : public CommonTest {};

namespace {
// PrimitiveType_Stack: src/ops/populate/stack_populate.cc
OpParameter *CreateParameter(int axis) {
  auto *param = test::CreateParameter<StackParameter>(schema::PrimitiveType_Stack);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

// stack test cases
TEST_F(TestOpenCL_Stack, input2_ndim1_axis1) {
  constexpr int INPUT_NUM = 2;
  int axis = 1;
  std::vector<int> input_shapes[INPUT_NUM] = {{10}, {10}};
  std::vector<int> output_shape = {10, 2};
  float input_datas[INPUT_NUM][10] = {{0.75, 0.06, 0.74, 0.30, 0.9, 0.59, 0.03, 0.37, 0.13, 0.47},
                                      {0.5, 0.6, 0.74, 0.23, 0.46, 0.69, 0.13, 0.47, 0.59, 0.03}};
  float output_data[] = {0.75, 0.5,  0.06, 0.6,  0.74, 0.74, 0.30, 0.23, 0.9,  0.46,
                         0.59, 0.69, 0.03, 0.13, 0.37, 0.47, 0.13, 0.59, 0.47, 0.03};

  for (auto fp16_enable : {true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shapes[0], input_datas[0], VAR}, {input_shapes[1], input_datas[1], VAR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestOpenCL_Stack, input2_ndim2_axis1) {
  constexpr int INPUT_NUM = 2;
  int axis = 1;
  std::vector<int> input_shapes[INPUT_NUM] = {{3, 4}, {3, 4}};
  std::vector<int> output_shape = {3, 2, 4};
  float input_datas[INPUT_NUM][12] = {
    {1.317, -2.094, -1.892, -0.4612, -0.884, -0.524, 0.4504, 0.0284, 3.227, -0.4673, -1.115, -0.1572},
    {-0.0677, -1.289, 0.0685, 0.889, 0.8145, 1.6455, 0.6587, -0.236, 0.3625, 0.7393, -1.393, 0.2534}};
  float output_data[] = {1.317,  -2.094,  -1.892, -0.4612, -0.0677, -1.289, 0.0685, 0.889,
                         -0.884, -0.524,  0.4504, 0.0284,  0.8145,  1.6455, 0.6587, -0.236,
                         3.227,  -0.4673, -1.115, -0.1572, 0.3625,  0.7393, -1.393, 0.2534};

  for (auto fp16_enable : {true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shapes[0], input_datas[0], VAR}, {input_shapes[1], input_datas[1], VAR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestOpenCL_Stack, input2_ndim3_axis1) {
  constexpr int INPUT_NUM = 2;
  int axis = 1;
  std::vector<int> input_shapes[INPUT_NUM] = {{3, 4, 5}, {3, 4, 5}};
  std::vector<int> output_shape = {3, 2, 4, 5};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/stack/input2_ndim3_axis1/stackfp32_input1.bin";
  std::string input2Ppath = "./test_data/stack/input2_ndim3_axis1/stackfp32_input2.bin";
  std::string correctOutputPath = "./test_data/stack/input2_ndim3_axis1/stackfp32_output.bin";
  auto input_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shapes[0], input_data1, VAR}, {input_shapes[1], input_data2, VAR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestOpenCL_Stack, input2_ndim3_axis2) {
  constexpr int INPUT_NUM = 2;
  int axis = 2;
  std::vector<int> input_shapes[INPUT_NUM] = {{3, 4, 5}, {3, 4, 5}};
  std::vector<int> output_shape = {3, 4, 2, 5};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/stack/input2_ndim3_axis2/stackfp32_input1.bin";
  std::string input2Ppath = "./test_data/stack/input2_ndim3_axis2/stackfp32_input2.bin";
  std::string correctOutputPath = "./test_data/stack/input2_ndim3_axis2/stackfp32_output.bin";
  auto input_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shapes[0], input_data1, VAR}, {input_shapes[1], input_data2, VAR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestOpenCL_Stack, input2_ndim2_axis2) {
  constexpr int INPUT_NUM = 2;
  int axis = 2;
  std::vector<int> input_shapes[INPUT_NUM] = {{1, 96}, {1, 96}};
  std::vector<int> output_shape = {1, 96, 2};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/stack/input2_ndim2_axis2/stackfp32_input1.bin";
  std::string input2Ppath = "./test_data/stack/input2_ndim2_axis2/stackfp32_input2.bin";
  std::string correctOutputPath = "./test_data/stack/input2_ndim2_axis2/stackfp32_output.bin";
  auto input_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shapes[0], input_data1, VAR}, {input_shapes[1], input_data2, VAR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestOpenCL_Stack, input2_ndim3_axis3) {
  constexpr int INPUT_NUM = 2;
  int axis = 3;
  std::vector<int> input_shapes[INPUT_NUM] = {{3, 4, 6}, {3, 4, 6}};
  std::vector<int> output_shape = {3, 4, 6, 2};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/stack/input2_ndim3_axis3/stackfp32_input1.bin";
  std::string input2Ppath = "./test_data/stack/input2_ndim3_axis3/stackfp32_input2.bin";
  std::string correctOutputPath = "./test_data/stack/input2_ndim3_axis3/stackfp32_output.bin";
  auto input_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shapes[0], input_data1, VAR}, {input_shapes[1], input_data2, VAR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

}  // namespace mindspore::lite::opencl::test
