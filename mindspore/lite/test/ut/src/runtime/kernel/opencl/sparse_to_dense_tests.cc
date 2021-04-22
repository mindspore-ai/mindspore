/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ut/src/runtime/kernel/opencl/common.h"
#include "nnacl/sparse_to_dense_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_SparseToDense : public CommonTest {};
// Check and optimize
namespace {
// PrimitiveType_SparseToDense: src/ops/populate/sparse_to_dense_populate.cc
OpParameter *CreateParameter() {
  auto *param = test::CreateParameter<SparseToDenseParameter>(schema::PrimitiveType_SparseToDense);
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_SparseToDense, Dim2Shape3Vector) {
  std::vector<int> input_shape0 = {6, 3};
  std::vector<int> input_shape1 = {3};
  std::vector<int> input_shape2 = {6};
  std::vector<int> input_shape3 = {1};
  std::vector<int> output_shape = {6, 1, 10};
  float input_data0[] = {0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6};
  float input_data1[] = {6, 1, 10};
  float input_data2[] = {1, 2, 3, 4, 5, 6};
  float input_data3[] = {0};
  float output_data[] = {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter();
    TestMain({{input_shape0, input_data0, VAR},
              {input_shape1, input_data1, CONST_TENSOR},
              {input_shape2, input_data2, CONST_TENSOR},
              {input_shape3, input_data3, CONST_SCALAR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SparseToDense, Dim2Scalar) {
  std::vector<int> input_shape0 = {6, 2};
  std::vector<int> input_shape1 = {2};
  std::vector<int> input_shape2 = {1};
  std::vector<int> input_shape3 = {1};
  std::vector<int> output_shape = {6, 10};
  float input_data0[] = {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9};
  float input_data1[] = {6, 10};
  float input_data2[] = {6};
  float input_data3[] = {0};
  float output_data[] = {6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter();
    TestMain({{input_shape0, input_data0, VAR},
              {input_shape1, input_data1, CONST_TENSOR},
              {input_shape2, input_data2, CONST_SCALAR},
              {input_shape3, input_data3, CONST_SCALAR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SparseToDense, Dim2Vector) {
  std::vector<int> input_shape0 = {6, 2};
  std::vector<int> input_shape1 = {2};
  std::vector<int> input_shape2 = {6};
  std::vector<int> input_shape3 = {1};
  std::vector<int> output_shape = {6, 10};
  float input_data0[] = {0, 0, 1, 2, 2, 3, 3, 6, 4, 7, 5, 9};
  float input_data1[] = {6, 10};
  float input_data2[] = {1, 2, 3, 4, 5, 6};
  float input_data3[] = {0};
  float output_data[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter();
    TestMain({{input_shape0, input_data0, VAR},
              {input_shape1, input_data1, CONST_TENSOR},
              {input_shape2, input_data2, CONST_TENSOR},
              {input_shape3, input_data3, CONST_SCALAR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SparseToDense, Dim2Shape1Vector) {
  std::vector<int> input_shape0 = {6, 1};
  std::vector<int> input_shape1 = {1};
  std::vector<int> input_shape2 = {6};
  std::vector<int> input_shape3 = {1};
  std::vector<int> output_shape = {10};
  float input_data0[] = {0, 2, 3, 6, 7, 9};
  float input_data1[] = {10};
  float input_data2[] = {1, 2, 3, 4, 5, 6};
  float input_data3[] = {0};
  float output_data[] = {1, 0, 2, 3, 0, 0, 4, 5, 0, 6};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter();
    TestMain({{input_shape0, input_data0, VAR},
              {input_shape1, input_data1, CONST_TENSOR},
              {input_shape2, input_data2, CONST_TENSOR},
              {input_shape3, input_data3, CONST_SCALAR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SparseToDense, Dim2Shape1Scalar) {
  std::vector<int> input_shape0 = {7, 1};
  std::vector<int> input_shape1 = {1};
  std::vector<int> input_shape2 = {1};
  std::vector<int> input_shape3 = {1};
  std::vector<int> output_shape = {10};
  float input_data0[] = {0, 1, 2, 3, 4, 5, 9};
  float input_data1[] = {10};
  float input_data2[] = {6};
  float input_data3[] = {0};
  float output_data[] = {6, 6, 6, 6, 6, 6, 0, 0, 0, 6};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter();
    TestMain({{input_shape0, input_data0, VAR},
              {input_shape1, input_data1, CONST_TENSOR},
              {input_shape2, input_data2, CONST_SCALAR},
              {input_shape3, input_data3, CONST_SCALAR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SparseToDense, Dim1Scalar) {
  std::vector<int> input_shape0 = {6};
  std::vector<int> input_shape1 = {1};
  std::vector<int> input_shape2 = {1};
  std::vector<int> input_shape3 = {1};
  std::vector<int> output_shape = {10};
  float input_data0[] = {1, 3, 4, 5, 6, 7};
  float input_data1[] = {10};
  float input_data2[] = {1};
  float input_data3[] = {2};
  float output_data[] = {2, 1, 2, 1, 1, 1, 1, 1, 2, 2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter();
    TestMain({{input_shape0, input_data0, VAR},
              {input_shape1, input_data1, CONST_TENSOR},
              {input_shape2, input_data2, CONST_SCALAR},
              {input_shape3, input_data3, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_SparseToDense, Dim1Vector) {
  std::vector<int> input_shape0 = {6};
  std::vector<int> input_shape1 = {1};
  std::vector<int> input_shape2 = {6};
  std::vector<int> input_shape3 = {1};
  std::vector<int> output_shape = {10};
  float input_data0[] = {1, 3, 4, 5, 6, 7};
  float input_data1[] = {10};
  float input_data2[] = {1, 2, 3, 4, 5, 6};
  float input_data3[] = {2};
  float output_data[] = {2, 1, 2, 2, 3, 4, 5, 6, 2, 2};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter();
    TestMain({{input_shape0, input_data0, VAR},
              {input_shape1, input_data1, CONST_TENSOR},
              {input_shape2, input_data2, CONST_TENSOR},
              {input_shape3, input_data3, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
