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
#include "nnacl/arithmetic.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Arithmetic : public CommonTest {};

namespace {
// PrimitiveType_RealDiv
// PrimitiveType_LogicalAnd
// PrimitiveType_LogicalOr
// PrimitiveType_Equal
// PrimitiveType_Less
// PrimitiveType_Greater
// PrimitiveType_GreaterEqual
// PrimitiveType_NotEqual
// PrimitiveType_LessEqual
// PrimitiveType_Maximum
// PrimitiveType_Minimum
// PrimitiveType_FloorDiv
// PrimitiveType_FloorMod
// PrimitiveType_SquaredDifference: src/ops/populate/arithmetic_populate.cc
// PrimitiveType_Add:               src/ops/populate/add_populate.cc
// PrimitiveType_Sub:               src/ops/populate/sub_populate.cc
// PrimitiveType_Mul:               src/ops/populate/mul_populate.cc
// PrimitiveType_Div:               src/ops/populate/div_populate.cc
// PrimitiveType_Eltwise:           src/ops/populate/eltwise_populate.cc
// PrimitiveType_BiasAdd:           src/ops/populate/bias_add_populate.cc
OpParameter *CreateParameter(schema::PrimitiveType type, const std::vector<int> &input0_shape,
                             const std::vector<int> &input1_shape,
                             schema::ActivationType act_type = schema::ActivationType_NO_ACTIVATION) {
  auto *param = test::CreateParameter<ArithmeticParameter>(type);
  int input0_size = std::accumulate(input0_shape.begin(), input0_shape.end(), 1, std::multiplies<>());
  int input1_size = std::accumulate(input1_shape.begin(), input1_shape.end(), 1, std::multiplies<>());
  if (input0_size != input1_size) {
    param->broadcasting_ = true;
  }
  param->activation_type_ = act_type;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Arithmetic, ElementwiseAdd) {
  std::vector<int> input0_shape = {1, 2, 2, 3};
  std::vector<int> input1_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 2, 2, 3};
  float input0_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float input1_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float output_data[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::PrimitiveType_AddFusion, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Arithmetic, ScalarMul) {
  std::vector<int> input0_shape = {1, 2, 2, 3};
  std::vector<int> input1_shape = {1};
  std::vector<int> output_shape = {1, 2, 2, 3};
  float input0_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float input1_data[] = {2};
  float output_data[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::PrimitiveType_MulFusion, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Arithmetic, BroadcastSubReLU6) {
  std::vector<int> input0_shape = {1, 2, 2, 3};
  std::vector<int> input1_shape = {3};
  std::vector<int> output_shape = {1, 2, 2, 3};
  float input0_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float input1_data[] = {1, 2, 3};
  float output_data[] = {0, 0, 0, 3, 3, 3, 6, 6, 6, 6, 6, 6};
  for (auto fp16_enable : {false, true}) {
    auto *param =
      CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape, schema::ActivationType_RELU6);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Arithmetic, BroadcastSub2) {
  std::vector<int> input0_shape = {1, 3};
  std::vector<int> input1_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 2, 2, 3};
  float input0_data[] = {1, 2, 3};
  float input1_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float output_data[] = {0, 0, 0, -3, -3, -3, -6, -6, -6, -9, -9, -9};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Arithmetic, BroadcastSub3) {
  std::vector<int> input0_shape = {2, 3};
  std::vector<int> input1_shape = {2, 2, 2, 3};
  std::vector<int> output_shape = {2, 2, 2, 3};
  float input0_data[] = {1, 2, 3, 1, 2, 3};
  float input1_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float output_data[] = {0, 0, 0, -3, -3, -3, -6, -6, -6, -9, -9, -9, 0, 0, 0, -3, -3, -3, -6, -6, -6, -9, -9, -9};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::PrimitiveType_SubFusion, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Arithmetic, BroadcastFloorMod) {
  std::vector<int> input0_shape = {1, 1, 3, 4};
  std::vector<int> input1_shape = {1, 1, 1, 4};
  std::vector<int> output_shape = {1, 1, 3, 4};
  float input0_data[] = {1.1, -1.1, 3.123, -5.432, 0.1234, -0.0312, 12.1, 21.1, 9.1, 9.0, -100, 0.1};
  float input1_data[] = {1, 3, 2, 0.3};
  float output_data[] = {0.100000, 1.900000, 1.123000, 0.268000, 0.123400, 2.968800,
                         0.100000, 0.100000, 0.100000, 0.000000, 0.000000, 0.100000};
  for (auto fp16_enable : {true, false}) {
    auto *param = CreateParameter(schema::PrimitiveType_FloorMod, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-2 : 1e-6);
  }
}

TEST_F(TestOpenCL_Arithmetic, FloorMod) {
  std::vector<int> input0_shape = {1, 1, 3, 4};
  std::vector<int> input1_shape = {1, 1, 3, 4};
  std::vector<int> output_shape = {1, 1, 3, 4};
  float input0_data[] = {1.1, -1.1, 3.123, -5.432, 0.1234, -0.0312, 12.1, 21.1, 9.1, 9.0, -100, 0.1};
  float input1_data[] = {1, 3, 2, 0.3, 1, 3, 2, 0.3, 1, 3, 2, 0.3};
  float output_data[] = {0.100000, 1.900000, 1.123000, 0.268000, 0.123400, 2.968800,
                         0.100000, 0.100000, 0.100000, 0.000000, 0.000000, 0.100000};
  for (auto fp16_enable : {true, false}) {
    auto *param = CreateParameter(schema::PrimitiveType_FloorMod, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-2 : 1e-6);
  }
}

TEST_F(TestOpenCL_Arithmetic, FloorModFile) {
  std::vector<int> input0_shape = {1, 3, 4, 5};
  std::vector<int> input1_shape = {1, 3, 4, 5};
  std::vector<int> output_shape = {1, 3, 4, 5};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/FloodModfp32_input1.bin";
  std::string input2Ppath = "./test_data/FloodModfp32_input2.bin";
  std::string correctOutputPath = "./test_data/FloodModfp32_output.bin";
  auto input0_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input1_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  for (auto fp16_enable : {true}) {
    auto *param = CreateParameter(schema::PrimitiveType_FloorMod, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-2 : 1e-7);
  }
}

TEST_F(TestOpenCL_Arithmetic, SquaredDifference) {
  std::vector<int> input0_shape = {1, 512, 1, 5};
  std::vector<int> input1_shape = {1, 1, 1, 5};
  std::vector<int> output_shape = {1, 512, 1, 5};
  size_t input1_size, input2_size, output_size;
  std::string input1Ppath = "./test_data/SquaredDifferencefp32_input1.bin";
  std::string input2Ppath = "./test_data/SquaredDifferencefp32_input2.bin";
  std::string correctOutputPath = "./test_data/SquaredDifferencefp32_output.bin";
  auto input0_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1Ppath.c_str(), &input1_size));
  auto input1_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input2Ppath.c_str(), &input2_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));

  for (auto fp16_enable : {true}) {
    auto *param = CreateParameter(schema::PrimitiveType_SquaredDifference, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable, fp16_enable ? 1e-2 : 1e-9);
  }
}

TEST_F(TestOpenCL_Arithmetic, ElementwiseDiv) {
  std::vector<int> input0_shape = {1, 2, 2, 3};
  std::vector<int> input1_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 2, 2, 3};
  float input0_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float input1_data[] = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2};
  float output_data[] = {1, 2, 3, 2, 2.5, 3, 7, 8, 9, 5, 5.5, 6};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::PrimitiveType_DivFusion, input0_shape, input1_shape);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, CONST_TENSOR}}, {output_shape, output_data},
             param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
