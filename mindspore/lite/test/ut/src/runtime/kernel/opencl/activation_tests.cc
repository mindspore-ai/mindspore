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
#include "nnacl/fp32/activation_fp32.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Activation : public CommonTest {};

namespace {
// PrimitiveType_Activation: src/ops/populate/activation_populate.cc
OpParameter *CreateParameter(schema::ActivationType act_type) {
  auto *param = test::CreateParameter<ActivationParameter>(schema::PrimitiveType_Activation);
  param->type_ = act_type;
  param->alpha_ = 0.0f;
  param->min_val_ = 0.0f;
  param->max_val_ = 0.0f;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Activation, RELU) {
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {-1, 1, 2, 3, -1, -2, 3, -4, 5, -6, 7, 9};
  float output_data[] = {0, 1, 2, 3, 0, 0, 3, 0, 5, 0, 7, 9};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::ActivationType_RELU);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Activation, RELU6) {
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {-1, 1, 2, 3, -1, -2, 3, -4, 5, -6, 7, 9};
  float output_data[] = {0, 1, 2, 3, 0, 0, 3, 0, 5, 0, 6, 6};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::ActivationType_RELU6);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Activation, HSIGMOID) {
  std::vector<int> input_shape = {2, 10, 1, 4};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {2.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5};
  float output_data[] = {0.9166667, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0,
                         1,         1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                         0,         1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::ActivationType_HSIGMOID);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 1e-3 : 1e-4);
  }
}

TEST_F(TestOpenCL_Activation, HSWISH) {
  std::vector<int> input_shape = {2, 10, 1, 4};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {2.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5,
                        7.5, 6, -7.4, -3.5, 5.9, 6.5, -8, 7.4, 5.9, 6.5, -8, 7.4, 7.5, 6, -7.4, -3.5};
  float output_data[] = {2.29166667, 6, 0, 0, 5.9, 6.5, 0, 7.4, 5.9, 6.5, 0, 7.4, 7.5, 6, 0, 0,
                         7.5,        6, 0, 0, 5.9, 6.5, 0, 7.4, 5.9, 6.5, 0, 7.4, 7.5, 6, 0, 0,
                         7.5,        6, 0, 0, 5.9, 6.5, 0, 7.4, 5.9, 6.5, 0, 7.4, 7.5, 6, 0, 0,
                         7.5,        6, 0, 0, 5.9, 6.5, 0, 7.4, 5.9, 6.5, 0, 7.4, 7.5, 6, 0, 0,
                         7.5,        6, 0, 0, 5.9, 6.5, 0, 7.4, 5.9, 6.5, 0, 7.4, 7.5, 6, 0, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::ActivationType_HSWISH);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 1e-2 : 1e-4);
  }
}

}  // namespace mindspore::lite::opencl::test
