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
#include "nnacl/gather_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Gather : public CommonTest {};

namespace {
// PrimitiveType_Gather: src/ops/populate/gather_populate.cc
OpParameter *CreateParameter(int axis) {
  auto *param = test::CreateParameter<GatherParameter>(schema::PrimitiveType_Gather);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Gather, Axis0) {
  int axis = 0;
  std::vector<int> input_shape = {10};
  std::vector<int> indices_shape = {5};
  std::vector<int> output_shape = {5};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int32_t indices[] = {1, 3, 5, 7, 9};
  float output_data[] = {1, 3, 5, 7, 9};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain(
      {{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices, CONST_TENSOR, kNumberTypeInt32}},
      {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Gather, Axis0ConstTensor) {
  int axis = 0;
  std::vector<int> input_shape = {10};
  std::vector<int> indices_shape = {1};
  std::vector<int> output_shape = {1};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int32_t indices[] = {1};
  float output_data[] = {1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain(
      {{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices, CONST_TENSOR, kNumberTypeInt32}},
      {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Gather, Axis0_Tensor) {
  int axis = 0;
  std::vector<int> input_shape = {10};
  std::vector<int> indices_shape = {1};
  std::vector<int> output_shape = {1};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int32_t indices[] = {1};
  float output_data[] = {1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices, VAR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestOpenCL_Gather, Axis1) {
  int axis = 1;
  std::vector<int> input_shape = {1, 5, 4, 4};
  std::vector<int> indices_shape = {2};
  std::vector<int> output_shape = {1, 2, 4, 4};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  float output_data[] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

  int32_t indices_int32[] = {1, 3};
  int64_t indices_int64[] = {1, 3};
  float32_t indices_fp32[] = {1, 3};
  float16_t indices_fp16[] = {1, 3};
  TypeId data_types[] = {kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat16};
  void *indices_datas[] = {indices_int32, indices_int64, indices_fp32, indices_fp16};

  for (int i = 0; i < 1; ++i) {
    for (auto fp16_enable : {false, true}) {
      auto *param = CreateParameter(axis);
      TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
                {indices_shape, indices_datas[i], CONST_TENSOR, data_types[i]}},
               {output_shape, output_data}, param, fp16_enable);
    }
  }
}

TEST_F(TestOpenCL_Gather, Axis1_intensor1) {
  int axis = 1;
  std::vector<int> input_shape = {1, 5, 4, 4};
  std::vector<int> indices_shape = {2};
  std::vector<int> output_shape = {1, 2, 4, 4};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  float output_data[] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

  int32_t indices_int32[] = {1, 3};
  int64_t indices_int64[] = {1, 3};
  float32_t indices_fp32[] = {1, 3};
  float16_t indices_fp16[] = {1, 3};
  TypeId data_types[] = {kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat32, kNumberTypeFloat16};
  void *indices_datas[] = {indices_int32, indices_int64, indices_fp32, indices_fp16};

  for (int i = 0; i < 1; ++i) {
    for (auto fp16_enable : {false}) {
      auto *param = CreateParameter(axis);
      TestMain(
        {{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices_datas[i], VAR, data_types[i]}},
        {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-3 : 1e-9);
    }
  }
}

TEST_F(TestOpenCL_Gather, Axis2) {
  int axis = 2;
  std::vector<int> input_shape = {1, 5, 4, 4};
  std::vector<int> indices_shape = {2};
  std::vector<int> output_shape = {1, 5, 2, 4};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  int32_t indices[] = {1, 3};
  float output_data[] = {4,  5,  6,  7,  12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39,
                         44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63, 68, 69, 70, 71, 76, 77, 78, 79};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain(
      {{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices, CONST_TENSOR, kNumberTypeInt32}},
      {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Gather, Axis2_intensor1) {
  int axis = 2;
  std::vector<int> input_shape = {1, 5, 4, 4};
  std::vector<int> indices_shape = {2};
  std::vector<int> output_shape = {1, 5, 2, 4};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  int32_t indices[] = {1, 3};
  float output_data[] = {4,  5,  6,  7,  12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39,
                         44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63, 68, 69, 70, 71, 76, 77, 78, 79};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices, VAR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Gather, Axis3) {
  int axis = 3;
  std::vector<int> input_shape = {1, 5, 4, 4};
  std::vector<int> indices_shape = {2};
  std::vector<int> output_shape = {1, 5, 4, 2};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  int32_t indices[] = {1, 3};
  float output_data[] = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
                         41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain(
      {{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices, CONST_TENSOR, kNumberTypeInt32}},
      {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Gather, Axis3_intensor1) {
  int axis = 3;
  std::vector<int> input_shape = {1, 5, 4, 4};
  std::vector<int> indices_shape = {2};
  std::vector<int> output_shape = {1, 5, 4, 2};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
  int32_t indices[] = {1, 3};
  float output_data[] = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
                         41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32}, {indices_shape, indices, VAR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
