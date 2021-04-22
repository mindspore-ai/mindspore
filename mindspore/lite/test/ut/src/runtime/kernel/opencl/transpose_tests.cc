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
#include "nnacl/transpose.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Transpose : public CommonTest {};

namespace {
// PrimitiveType_Transpose: src/ops/populate/transpose_populate.cc
//                          src/ops/populate/nchw2nhwc_populate.cc
//                          src/ops/populate/nhwc2nchw_populate.cc
OpParameter *CreateParameter(const std::vector<int> &perm) {
  auto *param = test::CreateParameter<TransposeParameter>(schema::PrimitiveType_Transpose);
  param->num_axes_ = perm.size();
  for (int i = 0; i < perm.size(); ++i) {
    param->perm_[i] = perm[i];
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Transpose, NHWC2NCHW) {
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> perm = {0, 3, 1, 2};
  std::vector<int> output_shape;
  for (int axis : perm) {
    output_shape.push_back(input_shape[axis]);
  }
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(perm);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(perm.size())}, {perm.data()}, CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Transpose, NCHW2NHWC) {
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> perm = {0, 2, 3, 1};
  std::vector<int> output_shape;
  for (int axis : perm) {
    output_shape.push_back(input_shape[axis]);
  }
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(perm);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(perm.size())}, {perm.data()}, CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Transpose, NHWC2NWHC) {
  std::vector<int> input_shape = {1, 2, 3, 4};
  std::vector<int> perm = {0, 2, 1, 3};
  std::vector<int> output_shape;
  for (int axis : perm) {
    output_shape.push_back(input_shape[axis]);
  }
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  float output_data[] = {0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(perm);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(perm.size())}, {perm.data()}, CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Transpose, NWC2CWN) {
  std::vector<int> input_shape = {1, 2, 3};
  std::vector<int> perm = {2, 1, 0};
  std::vector<int> output_shape;
  for (int axis : perm) {
    output_shape.push_back(input_shape[axis]);
  }
  float input_data[] = {0, 1, 2, 3, 4, 5};
  float output_data[] = {0, 3, 1, 4, 2, 5};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(perm);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(perm.size())}, {perm.data()}, CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Transpose, NWC2WNC) {
  std::vector<int> input_shape = {2, 3, 5};
  std::vector<int> perm = {1, 0, 2};
  std::vector<int> output_shape;
  for (int axis : perm) {
    output_shape.push_back(input_shape[axis]);
  }
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  float output_data[] = {0,  1,  2,  3,  4,  15, 16, 17, 18, 19, 5,  6,  7,  8,  9,
                         20, 21, 22, 23, 24, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(perm);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(perm.size())}, {perm.data()}, CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
