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
#include "nnacl/matmul_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_MatMul : public CommonTest {};

namespace {
// PrimitiveType_MatMul: src/ops/populate/matmul_populate.cc
OpParameter *CreateParameter(bool a_transpose = false, bool b_transpose = true) {
  auto *param = test::CreateParameter<MatMulParameter>(schema::PrimitiveType_MatMul);
  param->a_transpose_ = a_transpose;
  param->b_transpose_ = b_transpose;
  param->has_bias_ = false;
  param->act_type_ = ActType_No;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_MatMul, 2Dfile) {
  std::vector<int> input_shape = {64, 64};
  std::vector<int> output_shape = {64, 64};
  std::vector<int> weight_shape = {64, 64};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/matmulfp32_input1.bin";
  std::string input2Ppath = "./test_data/matmulfp32_input2.bin";
  std::string correctOutputPath = "./test_data/matmulfp32_output.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto weight_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(false, false);
    TestMain({{input_shape, input_data, VAR}, {weight_shape, weight_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-3 : 1e-3);
  }
}

TEST_F(TestOpenCL_MatMul, 2D) {
  int ci = 5;
  int co = 3;
  std::vector<int> input_shape = {1, ci};
  std::vector<int> output_shape = {1, co};
  std::vector<int> weight_shape = {co, ci};
  float input_data[] = {0, 1, 2, 3, 4};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float output_data[] = {10, 10, 10};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter();
    TestMain({{input_shape, input_data, VAR}, {weight_shape, weight_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_MatMul, 4D) {
  int a = 1;
  int b = 2;
  int m = 2;
  int ci = 5;
  int co = 3;
  std::vector<int> input_shape = {a, b, m, ci};
  std::vector<int> output_shape = {a, b, m, co};
  std::vector<int> weight_shape = {a, b, co, ci};
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float weight_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
  float output_data[] = {15, 40, 65, 15, 40, 65, 90, 115, 140, 90, 115, 140};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter();
    TestMain({{input_shape, input_data, VAR}, {weight_shape, weight_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_MatMul, 3D) {
  int a = 2;
  int m = 2;
  int ci = 5;
  int co = 3;
  std::vector<int> input_shape = {a, m, ci};
  std::vector<int> output_shape = {a, m, co};
  std::vector<int> weight_shape = {a, co, ci};
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float weight_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
  float output_data[] = {15, 40, 65, 15, 40, 65, 90, 115, 140, 90, 115, 140};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter();
    TestMain({{input_shape, input_data, VAR}, {weight_shape, weight_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_MatMul, ActWeightTransposeB3D) {
  int a = 2;
  int m = 2;
  int ci = 5;
  int co = 3;
  std::vector<int> input_shape = {a, m, ci};
  std::vector<int> output_shape = {a, m, co};
  std::vector<int> weight_shape = {a, co, ci};
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float weight_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
  float output_data[] = {15, 40, 65, 15, 40, 65, 90, 115, 140, 90, 115, 140};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter();
    TestMain({{input_shape, input_data, VAR}, {weight_shape, weight_data, VAR}}, {output_shape, output_data}, param,
             fp16_enable);
  }
}

TEST_F(TestOpenCL_MatMul, ActWeight3D) {
  int a = 2;
  int m = 2;
  int ci = 5;
  int co = 3;
  std::vector<int> input_shape = {a, m, ci};
  std::vector<int> output_shape = {a, m, co};
  std::vector<int> weight_shape = {a, ci, co};
  float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float weight_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
  float output_data[] = {35, 40, 45, 35, 40, 45, 110, 115, 120, 110, 115, 120};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(false, false);
    TestMain({{input_shape, input_data, VAR}, {weight_shape, weight_data, VAR}}, {output_shape, output_data}, param,
             fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
