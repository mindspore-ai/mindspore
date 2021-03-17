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
#include "nnacl/reduce_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Reduce : public CommonTest {};

namespace {
// PrimitiveType_Reduce:    src/ops/populate/reduce_populate.cc
// PrimitiveType_Mean:      src/ops/populate/mean_populate.cc
OpParameter *CreateParameter(const std::vector<int> &axis, schema::ReduceMode mode, bool keep_dims) {
  auto *param = test::CreateParameter<ReduceParameter>(schema::PrimitiveType_ReduceFusion);
  param->keep_dims_ = keep_dims;
  param->reduce_to_end_ = false;
  param->coeff = 0.f;
  param->num_axes_ = axis.size();
  param->mode_ = mode;
  for (int i = 0; i < axis.size(); ++i) {
    param->axes_[i] = axis[i];
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Reduce, Mean) {
  std::vector<int> axis = {1, 2};
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 3};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {4.5, 5.5, 6.5f};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis, schema::ReduceMode_ReduceMean, false);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(axis.size())}, axis.data(), CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Reduce, Sum) {
  std::vector<int> axis = {1, 2};
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = {1, 3};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {18, 22, 26};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis, schema::ReduceMode_ReduceSum, false);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(axis.size())}, axis.data(), CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Reduce, MeanWC) {
  std::vector<int> axis = {2, 3};
  std::vector<int> input_shape = {1, 3, 2, 2};
  std::vector<int> output_shape = {1, 3, 1, 1};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {1.5, 5.5, 9.5f};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis, schema::ReduceMode_ReduceMean, true);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(axis.size())}, axis.data(), CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Reduce, SumWC) {
  std::vector<int> axis = {2, 3};
  std::vector<int> input_shape = {1, 3, 2, 2};
  std::vector<int> output_shape = {1, 3, 1, 1};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {6, 22, 38};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis, schema::ReduceMode_ReduceSum, true);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(axis.size())}, axis.data(), CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Reduce, MeanC) {
  std::vector<int> axis = {3};
  std::vector<int> input_shape = {1, 3, 2, 2};
  std::vector<int> output_shape = {1, 3, 2, 1};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  float output_data[] = {0.5, 2.5, 4.5, 6.5, 8.5, 10.5};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis, schema::ReduceMode_ReduceMean, true);
    TestMain({{input_shape, input_data, VAR, kNumberTypeFloat32},
              {{static_cast<int>(axis.size())}, axis.data(), CONST_TENSOR, kNumberTypeInt32}},
             {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
