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
#include "nnacl/reshape_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Reshape : public CommonTest {};

namespace {
// PrimitiveType_Reshape: src/ops/populate/reshape_populate.cc
OpParameter *CreateParameter() {
  auto *param = test::CreateParameter<ReshapeParameter>(schema::PrimitiveType_Reshape);
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Reshape, 4D_2D_test0) {
  std::vector<int> shape_in = {1, 1, 1, 7};
  std::vector<int> shape_out = {1, 7};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6};
  float output_data[] = {0, 1, 2, 3, 4, 5, 6};
  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 4D_4D_test0) {
  std::vector<int> shape_in = {1, 2, 2, 3};
  std::vector<int> shape_out = {1, 1, 4, 3};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 4D_2D_test1) {
  std::vector<int> shape_in = {1, 2, 2, 4};
  std::vector<int> shape_out = {4, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float output_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 4D_4D_test1) {
  std::vector<int> shape_in = {1, 4, 2, 3};
  std::vector<int> shape_out = {1, 3, 2, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  float output_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 4D_4D_test2) {
  std::vector<int> shape_in = {1, 2, 2, 5};
  std::vector<int> shape_out = {1, 1, 5, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  float output_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 4D_4D_test3) {
  std::vector<int> shape_in = {1, 4, 2, 5};
  std::vector<int> shape_out = {1, 2, 5, 4};
  float input_data[] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
  };
  float output_data[] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
  };
  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 4D_4D_test4) {
  std::vector<int> shape_in = {1, 5, 5, 8};
  std::vector<int> shape_out = {8, 1, 5, 5};
  float input_data[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
    46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,
    69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
    92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
    115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
    138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199};
  float output_data[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
    46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,
    69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
    92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
    115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
    138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
    184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199};

  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 4D_4D_test5) {
  std::vector<int> shape_in = {1, 3, 2, 5};
  std::vector<int> shape_out = {1, 5, 2, 3};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  float output_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}

TEST_F(TestOpenCL_Reshape, 3D_2D_test6) {
  std::vector<int> shape_in = {5, 3, 8};
  std::vector<int> shape_out = {8, 15};
  float input_data[] = {
    0,  1,  2,  3,  4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
    24, 25, 26, 27, 28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
    48, 49, 50, 51, 52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
    72, 73, 74, 75, 76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
  float output_data[] = {
    0,  1,  2,  3,  4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
    24, 25, 26, 27, 28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
    48, 49, 50, 51, 52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
    72, 73, 74, 75, 76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};

  for (auto fp16_enable : {false, true}) {
    TestMain({{shape_in, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(shape_out.size())}, shape_out.data(), CONST_TENSOR, kNumberTypeInt32}},
             {shape_out, output_data}, CreateParameter(), fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
