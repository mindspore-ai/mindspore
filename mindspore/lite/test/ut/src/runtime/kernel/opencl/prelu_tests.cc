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
#include "mindspore/lite/nnacl/prelu_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_PRrelu : public CommonTest {};

namespace {
// PrimitiveType_PReLU: src/ops/populate/p_relu_populate.cc
OpParameter *CreateParameter() {
  auto *param = test::CreateParameter<PReluParameter>(schema::PrimitiveType_PReLUFusion);
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_PRrelu, testcase1) {
  std::vector<int> input_shape1 = {1, 4, 5, 6};
  std::vector<int> input_shape2 = {1};
  std::vector<int> output_shape = {1, 4, 5, 6};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/PRRelufp32_input1.bin";
  std::string input2Ppath = "./test_data/PRRelufp32_input2.bin";
  std::string correctOutputPath = "./test_data/PRRelufp32fp32_output.bin";
  auto input_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  for (auto fp16_enable : {true}) {
    auto *param = CreateParameter();
    TestMain({{input_shape1, input_data1, VAR}, {input_shape2, input_data2, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-2 : 1e-9);
  }
}

TEST_F(TestOpenCL_PRrelu, testcase2) {
  std::vector<int> input_shape1 = {1, 4, 5, 6};
  std::vector<int> input_shape2 = {1, 1, 1, 6};
  std::vector<int> output_shape = {1, 4, 5, 6};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/PRRelufp32_input1.bin";
  std::string input2Ppath = "./test_data/PRRelufp32_input2.bin";
  std::string correctOutputPath = "./test_data/PRRelufp32fp32_output.bin";
  auto input_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  for (auto fp16_enable : {true}) {
    auto *param = CreateParameter();
    TestMain({{input_shape1, input_data1, VAR}, {input_shape2, input_data2, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-2 : 1e-9);
  }
}
}  // namespace mindspore::lite::opencl::test
