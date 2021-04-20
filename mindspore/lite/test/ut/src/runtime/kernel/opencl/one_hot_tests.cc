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
#include "nnacl/fp32/one_hot_fp32.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_OneHot : public CommonTest {};

namespace {
// PrimitiveType_OneHot: src/ops/populate/one_hot_populate.cc
OpParameter *CreateParameter(int axis) {
  auto *param = test::CreateParameter<OneHotParameter>(schema::PrimitiveType_OneHot);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_OneHot, OneHot4DAxis3Fp32) {
  int depth = 4;
  int axis = -1;
  float on_value = 1;
  float off_value = -1;

  std::vector<int> input_shape = {1, 2, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {3, 4, -1, 2};
  float output_data[] = {-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1};
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis3T2Fp32) {
  int depth = 5;
  int axis = -1;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {1, 2, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {-1, 3, 4, 5};
  float output_data[] = {-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis3T3Fp32) {
  int depth = 9;
  int axis = -1;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {1, 2, 3};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {4, 9, 8, 9, 1, 8};
  float output_data[] = {-1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis3T4Fp32) {
  int depth = 6;
  int axis = -1;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {1, 2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {2, 4, 0, 6, 1, 6, 2, 2, 4, 5};
  float output_data[] = {-1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, 1,  -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1,
                         -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, 1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis2Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 2;
  float off_value = 0;
  std::vector<int> input_shape = {1, 2, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {2, 3, 0, 3};
  float output_data[] = {0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis2T2Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 2;
  float off_value = 0;
  std::vector<int> input_shape = {1, 6, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {1, 1, 1, 0, 1, 1, 4, -1, 4, 4, -1, 1};
  float output_data[] = {0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis2T3Fp32) {
  int depth = 1;
  int axis = 2;
  float on_value = 2;
  float off_value = 0;
  std::vector<int> input_shape = {1, 2, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {-1, 1, -1, 0};
  float output_data[] = {0, 0, 0, 2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis2T4Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {1, 2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {4, 0, -1, 2, 5, 4, -1, 4, 4, 4};
  float output_data[] = {-1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1,
                         -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, 1,  1,  1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis1T1Fp32) {
  int depth = 1;
  int axis = 1;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {1, 6, 6};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {0,  -1, 1, 0, -1, -1, 0, 0,  -1, 1, 0, -1, -1, 1, 1, -1, 1, 1,
                      -1, 1,  1, 1, -1, 0,  0, -1, 0,  0, 1, 1,  1,  1, 0, 0,  0, -1};
  float output_data[] = {2,  -2, -2, 2,  -2, -2, 2, 2,  -2, -2, 2,  -2, -2, -2, -2, -2, -2, -2,
                         -2, -2, -2, -2, -2, 2,  2, -2, 2,  2,  -2, -2, -2, -2, 2,  2,  2,  -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis1T2Fp32) {
  int depth = 4;
  int axis = 1;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {1, 2, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {-1, 1, 1, 2};
  float output_data[] = {-2, -2, -2, -2, -2, 2, 2, -2, -2, -2, -2, 2, -2, -2, -2, -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis1T3Fp32) {
  int depth = 5;
  int axis = 1;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {1, 2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {3, 5, 2, 0, 2, 2, -1, 0, 4, 3};
  float output_data[] = {-1, -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  -1, -1, -1,
                         -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis0Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {1, 2, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {4, 0, 3, 3};
  float output_data[] = {-2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 2, 2, 2, -2, -2, -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {1, 2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {2, 4, 4, 3, 5, 0, 3, 3, -1, 2};
  float output_data[] = {-1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, 1,
                         -1, -1, 1,  1,  -1, -1, -1, 1,  1,  -1, -1, -1, -1, -1, -1, -1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot4DAxis0T3Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {2, 2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {0, 3, 2, 0, 0, 3, 4, 1, 5, 1, 4, -1, 3, 3, 1, 1, 4, 2, 2, 4};
  float output_data[] = {1,  -1, -1, 1,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, 1,  -1, 1,  -1, -1, -1, -1, 1,  1,  -1, -1, -1, -1,
                         -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  1,  -1,
                         -1, 1,  -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, 1,  1,  -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, -1, -1, 1,  -1, -1, 1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot3DAxis0Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {2, 3};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {4, 4, 3, 2, -1, 5};
  float output_data[] = {-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
                         2,  -2, -2, -2, -2, 2,  -2, -2, -2, 2,  2,  -2, -2, -2, -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot3DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {4, 2, 2, 3, -1, 5, 2, 4, 5, -1};
  float output_data[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, 1,  1,  -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, 1,
                         -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, 1,  -1, -1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot3DAxis1Fp32) {
  int depth = 5;
  int axis = 1;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {2, 3};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {0, 0, 0, 0, 4, -1};
  float output_data[] = {2, 2,  2,  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
                         2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 2,  -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot3DAxis1T2Fp32) {
  int depth = 5;
  int axis = 1;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {1, -1, 3, 2, 5, 5, 4, 5, 0, -1};
  float output_data[] = {-1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1,
                         1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot3DAxis2Fp32) {
  int depth = 4;
  int axis = 2;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {2, 2};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {0, 3, 4, 2};
  float output_data[] = {2, -2, -2, -2, -2, -2, -2, 2, -2, -2, -2, -2, -2, -2, 2, -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot3DAxis2T2Fp32) {
  int depth = 5;
  int axis = 2;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {2, 5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {0, -1, 2, -1, 5, 4, 2, -1, 4, -1};
  float output_data[] = {1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, 1,  -1,
                         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot2DAxis0Fp32) {
  int depth = 3;
  int axis = 0;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {3};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {2, 1, 3};
  float output_data[] = {-2, -2, -2, -2, 2, -2, 2, -2, -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot2DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {2, 2, 0, 0, 4};
  float output_data[] = {-1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot2DAxis1Fp32) {
  int depth = 3;
  int axis = -1;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {3};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {1, 2, 0};
  float output_data[] = {-2, 2, -2, -2, -2, 2, 2, -2, -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot2DAxis1T2Fp32) {
  int depth = 5;
  int axis = -1;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {5};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {5, 4, 0, 4, -1};
  float output_data[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  1,  -1, -1,
                         -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot1DAxis0Fp32) {
  int depth = 3;
  int axis = -1;
  float on_value = 2;
  float off_value = -2;
  std::vector<int> input_shape = {};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {1};
  float output_data[] = {-2, 2, -2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_OneHot, OneHot1DAxis0T2Fp32) {
  int depth = 5;
  int axis = 0;
  float on_value = 1;
  float off_value = -1;
  std::vector<int> input_shape = {};
  std::vector<int> output_shape = input_shape;
  output_shape.insert(output_shape.begin() + (axis + input_shape.size() + 1) % (input_shape.size() + 1), depth);
  int input_data[] = {4};
  float output_data[] = {-1, -1, -1, -1, 1};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR, kNumberTypeInt32},
              {{}, &depth, CONST_SCALAR, kNumberTypeInt32},
              {{}, &on_value, CONST_SCALAR, kNumberTypeFloat32},
              {{}, &off_value, CONST_SCALAR, kNumberTypeFloat32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
