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
#include "nnacl/pad_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Pad : public CommonTest {};

namespace {
// PrimitiveType_Pad: src/ops/populate/pad_populate.cc
OpParameter *CreateParameter(const std::vector<int> &paddings, float constant_value) {
  auto *param = test::CreateParameter<PadParameter>(schema::PrimitiveType_PadFusion);
  param->pad_mode_ = schema::PaddingMode_CONSTANT;
  param->constant_value_ = constant_value;
  param->padding_length = MAX_PAD_SIZE;
  int size = paddings.size();
  for (size_t i = 0; i < MAX_PAD_SIZE - size; ++i) {
    param->paddings_[i] = 0;
  }
  for (size_t i = 0; i < size; i++) {
    param->paddings_[MAX_PAD_SIZE - size + i] = paddings[i];
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Pad, 1D) {
  float input_data[] = {1, 1, 1, 1};
  float output_data[] = {2, 2, 2, 1, 1, 1, 1, 2, 2};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({3, 2}, 2);
    TestMain({{{4}, input_data, VAR}}, {{9}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Pad, 2D) {
  float input_data[] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
  float output_data[] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 1,  1,  1,  1,  1,  10, 10,
                         10, 2,  2,  2,  2,  2,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({1, 1, 1, 2}, 10);
    TestMain({{{2, 5}, input_data, VAR}}, {{4, 8}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Pad, 4D) {
  float input_data[48] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                          32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
  float output_data[300] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({0, 0, 3, 3, 3, 3, 0, 0}, 0);
    TestMain({{{1, 4, 4, 3}, input_data, VAR}}, {{1, 10, 10, 3}, output_data}, param, fp16_enable);
  }

  float output_data1[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({0, 0, 3, 3, 3, 3, 0, 0}, 1);
    TestMain({{{1, 4, 4, 3}, input_data, VAR}}, {{1, 10, 10, 3}, output_data1}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Pad, test0) {
  std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<float>,
                         std::vector<int>, float>>
    cases = {
      {"SimpleConstTest",
       {1, 2, 2, 1},
       {3, 2, 4, 1},
       {1, 2, 3, 4},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {1, 1, 0, 0, 1, 1, 0, 0},
       0},
      {"SimpleConstImageStyleTest",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 1, 0, 0},
       0},
      {"SimpleConst1DTest", {2}, {5}, {2, 3}, {0, 2, 3, 0, 0}, {1, 2}, 0},
      {"SimpleDynamicTest",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 1, 0, 0},
       0},
      {"AdvancedConstTest",
       {1, 2, 3, 1},
       {2, 4, 6, 1},
       {1, 2, 3, 4, 5, 6},
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 2, 3, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {1, 0, 0, 2, 0, 3, 0, 0},
       0},
      {"AdvancedConstImageStyleTest",
       {1, 2, 3, 1},
       {1, 4, 7, 1},
       {1, 2, 3, 4, 5, 6},
       {0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 2, 1, 3, 0, 0},
       0},
      {"AdvancedDynamicTest",
       {1, 2, 3, 1},
       {1, 4, 7, 1},
       {1, 2, 3, 4, 5, 6},
       {0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 2, 1, 3, 0, 0},
       0},
      {"SimpleConstTestUint8",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 1, 0, 0},
       0},
      {"SimpleConstTestInt8",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 1, 0, 0},
       0},
      {"SimpleConstFloat32ValuedTestUint8",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4, 5, 5, 5, 5, 5},
       {0, 0, 1, 1, 1, 1, 0, 0},
       5},
      {"SimpleConstFloat32ValuedTestInt8",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4, 5, 5, 5, 5, 5},
       {0, 0, 1, 1, 1, 1, 0, 0},
       5},
      {"Simple4DConstFloat32ValuedTest",
       {1, 1, 2, 1},
       {2, 1, 2, 2},
       {3, 3},
       {3, 5, 3, 5, 5, 5, 5, 5},
       {0, 1, 0, 0, 0, 0, 0, 1},
       5},
      {"SimpleConstInt32ValuedTest",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4, 5, 5, 5, 5, 5},
       {0, 0, 1, 1, 1, 1, 0, 0},
       5},
      {"SimpleDynamicTest",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0},
       {0, 0, 1, 1, 1, 1, 0, 0},
       0},
      {"SimpleDynamicValuedTest",
       {1, 2, 2, 1},
       {1, 4, 4, 1},
       {1, 2, 3, 4},
       {5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4, 5, 5, 5, 5, 5},
       {0, 0, 1, 1, 1, 1, 0, 0},
       5},
      {"AdvancedConstTest",
       {1, 2, 3, 1},
       {1, 4, 7, 1},
       {1, 2, 3, 4, 5, 6},
       {0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 2, 1, 3, 0, 0},
       0},
      {"AdvancedDynamicTest",
       {1, 2, 3, 1},
       {1, 4, 7, 1},
       {1, 2, 3, 4, 5, 6},
       {0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {0, 0, 0, 2, 1, 3, 0, 0},
       0},
    };

  for (auto &case_ : cases) {
    auto &name = std::get<0>(case_);
    auto &input_shape = std::get<1>(case_);
    auto &output_shape = std::get<2>(case_);
    auto input_data = std::get<3>(case_).data();
    auto output_data = std::get<4>(case_).data();
    auto &paddings = std::get<5>(case_);
    auto constant_value = std::get<6>(case_);
    std::cout << name << std::endl;
    for (auto fp16_enable : {false, true}) {
      auto *param = CreateParameter(paddings, constant_value);
      TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
    }
  }
}

}  // namespace mindspore::lite::opencl::test
