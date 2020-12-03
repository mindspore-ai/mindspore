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

class TestOpenCL_FullConnection : public CommonTest {};

namespace {
// PrimitiveType_FullConnection: src/ops/populate/full_connection_populate.cc
OpParameter *CreateParameter(std::vector<int> *input_shape, std::vector<int> *weight_shape,
                             std::vector<int> *bias_shape, std::vector<int> *output_shape, int ndim, int ci, int co,
                             int n = 1, int h = 1, int w = 1, int in_n = 1) {
  auto *param = test::CreateParameter<MatMulParameter>(schema::PrimitiveType_FullConnection);
  param->a_transpose_ = false;
  param->b_transpose_ = true;
  param->has_bias_ = true;
  param->act_type_ = ActType_No;

  if (ndim == 2) {
    *input_shape = {1, ci};
    *output_shape = {1, co};
    *weight_shape = {co, ci};
    *bias_shape = {co};
  } else if (ndim == 4) {
    *input_shape = {n, h, w, ci};
    *output_shape = {n, co};
    *weight_shape = {co, h * w * ci};
    *bias_shape = {co};
  } else if (ndim == 3) {
    *input_shape = {in_n, w, ci};
    *output_shape = {n, co};
    *weight_shape = {co, in_n * w * ci / n};
    *bias_shape = {co};
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_FullConnection, 2D) {
  int ndim = 2;
  int ci = 5;
  int co = 3;
  float input_data[] = {0, 1, 2, 3, 4};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float bias_data[] = {1, 1, 1};
  float output_data[] = {11, 11, 11};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param = CreateParameter(&input_shape, &weight_shape, &bias_shape, &output_shape, ndim, ci, co);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_FullConnection, 4D) {
  int ndim = 4;
  int ci = 4;
  int co = 2;
  int n = 1;
  int h = 2;
  int w = 1;
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float bias_data[] = {1, 1};
  float output_data[] = {29, 29};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param = CreateParameter(&input_shape, &weight_shape, &bias_shape, &output_shape, ndim, ci, co, n, h, w);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_FullConnection, 3D) {
  int ndim = 3;
  int ci = 3;
  int co = 4;
  int n = 2;
  int h = 1;
  int w = 4;
  int in_n = 1;
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float bias_data[] = {1, 1, 1, 1};
  float output_data[] = {16, 16, 16, 16, 52, 52, 52, 52};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param = CreateParameter(&input_shape, &weight_shape, &bias_shape, &output_shape, ndim, ci, co, n, h, w, in_n);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_FullConnection, 3DWeightVar) {
  int ndim = 3;
  int ci = 6;
  int co = 4;
  int n = 2;
  int h = 1;
  int w = 2;
  int in_n = 1;
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float bias_data[] = {1, 1, 1, 1};
  float output_data[] = {16, 16, 16, 16, 52, 52, 52, 52};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param = CreateParameter(&input_shape, &weight_shape, &bias_shape, &output_shape, ndim, ci, co, n, h, w, in_n);
    TestMain({{input_shape, input_data, VAR}, {weight_shape, weight_data, VAR}, {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
