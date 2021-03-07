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
#include "nnacl/layer_norm_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_LayerNorm : public CommonTest {};

namespace {
// PrimitiveType_Stack: src/ops/populate/stack_populate.cc
OpParameter *CreateParameter(float epsilon, int begin_norm_axis_, int begin_param_axis_) {
  auto *param = test::CreateParameter<LayerNormParameter>(schema::PrimitiveType_LayerNormFusion);
  param->epsilon_ = epsilon;
  param->begin_norm_axis_ = begin_norm_axis_;
  param->begin_params_axis_ = begin_param_axis_;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_LayerNorm, test1) {
  float epsilon = 1e-5;
  int begin_norm_axis_ = 3;
  int begin_param_axis_ = 3;
  std::vector<int> normalizedShape = {5};
  std::vector<int> input_shape = {2, 3, 4, 5};
  std::vector<int> gamma_shape = {1, 1, 1, 5};
  std::vector<int> beta_shape = {1, 1, 1, 5};
  std::vector<int> output_shape = {2, 3, 4, 5};
  size_t input_size, gamma_size, beta_size, output_size;
  std::string inputPpath = "./test_data/layernormfp32_input.bin";
  std::string gammaPpath = "./test_data/gammafp32_input.bin";
  std::string betaPpath = "./test_data/betafp32_input.bin";
  std::string correctOutputPath = "./test_data/layernormfp32_output.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(inputPpath.c_str(), &input_size));
  auto gamma_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(gammaPpath.c_str(), &gamma_size));
  auto beta_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(betaPpath.c_str(), &beta_size));
  auto output_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(correctOutputPath.c_str(), &output_size));
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(epsilon, begin_norm_axis_, begin_param_axis_);
    TestMain(
      {{input_shape, input_data, VAR}, {gamma_shape, gamma_data, CONST_TENSOR}, {beta_shape, beta_data, CONST_TENSOR}},
      {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-3 : 1e-5);
  }
}
}  // namespace mindspore::lite::opencl::test
