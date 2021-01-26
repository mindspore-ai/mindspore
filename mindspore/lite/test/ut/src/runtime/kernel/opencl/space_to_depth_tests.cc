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
#include "mindspore/lite/nnacl/space_to_depth_parameter.h"
#include "mindspore/lite/nnacl/base/space_to_depth_base.h"
#include "nnacl/depth_to_space_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_SpaceToDepth : public CommonTest {};
class TestOpenCL_DepthToSpace : public CommonTest {};

namespace {
// PrimitiveType_SpaceToDepth: src/ops/populate/space_to_depth_populate.cc
OpParameter *CreateParameter(int block_size) {
  auto *param = test::CreateParameter<SpaceToDepthParameter>(schema::PrimitiveType_SpaceToDepth);
  param->block_size_ = block_size;
  return reinterpret_cast<OpParameter *>(param);
}

OpParameter *CreateDepthToSpaceParameter(int block_size) {
  auto *param = test::CreateParameter<DepthToSpaceParameter>(schema::PrimitiveType_DepthToSpace);
  param->block_size_ = block_size;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_SpaceToDepth, AlignTest1) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 2, 2, 4};
  std::vector<int> output_shape = {1, 1, 1, 16};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float output_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SpaceToDepth, AlignTest2) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 4, 4, 4};
  std::vector<int> output_shape = {1, 2, 2, 16};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  float output_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23, 8,  9,  10, 11, 12, 13,
                         14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51,
                         52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SpaceToDepth, AlignTest3) {
  int block_size = 3;
  std::vector<int> input_shape = {1, 6, 6, 4};
  std::vector<int> output_shape = {1, 2, 2, 36};
  float input_data[] = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
                        18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
                        36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
                        54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
                        72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                        90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
                        108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                        126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143};
  float output_data[] = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  24,  25,  26,  27,  28,  29,
                         30,  31,  32,  33,  34,  35,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                         12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  36,  37,  38,  39,  40,  41,
                         42,  43,  44,  45,  46,  47,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
                         72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  96,  97,  98,  99,  100, 101,
                         102, 103, 104, 105, 106, 107, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  108, 109, 110, 111, 112, 113,
                         114, 115, 116, 117, 118, 119, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SpaceToDepth, NotAlignTest1) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, 1, 1, 4};
  float input_data[] = {0, 1, 2, 3};
  float output_data[] = {0, 1, 2, 3};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SpaceToDepth, NotAlignTest2) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 1, 1, 12};
  float input_data[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
  };
  float output_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SpaceToDepth, NotAlignTest3) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 4, 4, 3};
  std::vector<int> output_shape = {1, 2, 2, 12};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
  float output_data[] = {0,  1,  2,  3,  4,  5,  12, 13, 14, 15, 16, 17, 6,  7,  8,  9,
                         10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 36, 37,
                         38, 39, 40, 41, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SpaceToDepth, NotAlignTest4) {
  int block_size = 3;
  std::vector<int> input_shape = {1, 6, 6, 6};
  std::vector<int> output_shape = {1, 2, 2, 54};
  float input_data[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
    66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
    88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215};
  float output_data[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  36,  37,  38,  39,
    40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  72,  73,  74,  75,  76,  77,  78,  79,
    80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
    30,  31,  32,  33,  34,  35,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
    70,  71,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, AlignTest1) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 2, 2, 16};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  float output_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23, 8,  9,  10, 11, 12, 13,
                         14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51,
                         52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, AlignTest2) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 1, 2, 16};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  float output_data[] = {0, 1, 2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
                         8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, AlignTest3) {
  int block_size = 3;
  std::vector<int> input_shape = {1, 1, 2, 36};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71};
  float output_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 36, 37, 38, 39, 40, 41,
                         42, 43, 44, 45, 46, 47, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 24, 25, 26, 27, 28, 29,
                         30, 31, 32, 33, 34, 35, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, AlignTest4) {
  int block_size = 4;
  std::vector<int> input_shape = {1, 1, 1, 64};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  float output_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                         44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, NotAlignTest1) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 3, 3, 8};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71};
  float output_data[] = {0,  1,  2,  3,  8,  9,  10, 11, 16, 17, 18, 19, 4,  5,  6,  7,  12, 13,
                         14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43,
                         28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 56, 57,
                         58, 59, 64, 65, 66, 67, 52, 53, 54, 55, 60, 61, 62, 63, 68, 69, 70, 71};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, NotAlignTest2) {
  int block_size = 3;
  std::vector<int> input_shape = {1, 3, 3, 9};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                        42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                        63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80};
  float output_data[] = {0,  1,  2,  9,  10, 11, 18, 19, 20, 3,  4,  5,  12, 13, 14, 21, 22, 23, 6,  7,  8,
                         15, 16, 17, 24, 25, 26, 27, 28, 29, 36, 37, 38, 45, 46, 47, 30, 31, 32, 39, 40, 41,
                         48, 49, 50, 33, 34, 35, 42, 43, 44, 51, 52, 53, 54, 55, 56, 63, 64, 65, 72, 73, 74,
                         57, 58, 59, 66, 67, 68, 75, 76, 77, 60, 61, 62, 69, 70, 71, 78, 79, 80};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, NotAlignTest3) {
  int block_size = 4;
  std::vector<int> input_shape = {1, 3, 2, 32};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
    66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
    88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191};
  float output_data[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   32,  33,  34,  35,  36,  37,  38,  39,  8,   9,   10,  11,  12,  13,
    14,  15,  40,  41,  42,  43,  44,  45,  46,  47,  16,  17,  18,  19,  20,  21,  22,  23,  48,  49,  50,  51,
    52,  53,  54,  55,  24,  25,  26,  27,  28,  29,  30,  31,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
    66,  67,  68,  69,  70,  71,  96,  97,  98,  99,  100, 101, 102, 103, 72,  73,  74,  75,  76,  77,  78,  79,
    104, 105, 106, 107, 108, 109, 110, 111, 80,  81,  82,  83,  84,  85,  86,  87,  112, 113, 114, 115, 116, 117,
    118, 119, 88,  89,  90,  91,  92,  93,  94,  95,  120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 160, 161, 162, 163, 164, 165, 166, 167, 136, 137, 138, 139, 140, 141, 142, 143, 168, 169,
    170, 171, 172, 173, 174, 175, 144, 145, 146, 147, 148, 149, 150, 151, 176, 177, 178, 179, 180, 181, 182, 183,
    152, 153, 154, 155, 156, 157, 158, 159, 184, 185, 186, 187, 188, 189, 190, 191};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_DepthToSpace, NotAlignTest4) {
  int block_size = 2;
  std::vector<int> input_shape = {1, 3, 4, 8};
  std::vector<int> output_shape = {1, input_shape[1] * block_size, input_shape[2] * block_size,
                                   input_shape[3] / (block_size * block_size)};
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  float output_data[] = {0,  1,  2,  3,  8,  9,  10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 4,  5,  6,  7,
                         12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 42, 43,
                         48, 49, 50, 51, 56, 57, 58, 59, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55,
                         60, 61, 62, 63, 64, 65, 66, 67, 72, 73, 74, 75, 80, 81, 82, 83, 88, 89, 90, 91,
                         68, 69, 70, 71, 76, 77, 78, 79, 84, 85, 86, 87, 92, 93, 94, 95};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateDepthToSpaceParameter(block_size);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
