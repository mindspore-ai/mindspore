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
#include "nnacl/split_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Split : public CommonTest {};

namespace {
// PrimitiveType_Split: src/ops/populate/split_populate.cc
OpParameter *CreateParameter(int split_dim_, int num_split_, std::vector<int> split_sizes_) {
  auto *param = test::CreateParameter<SplitParameter>(schema::PrimitiveType_Split);
  param->split_dim_ = split_dim_;
  param->num_split_ = num_split_;
  param->split_sizes_ = reinterpret_cast<int *>(malloc(param->num_split_ * sizeof(int)));
  for (int i = 0; i < param->num_split_; ++i) {
    param->split_sizes_[i] = split_sizes_[i];
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Split, input2_axis3) {
  std::vector<int> input_shape = {2, 2, 2, 12};
  std::vector<int> output_shape1 = {2, 2, 2, 6};
  std::vector<int> output_shape2 = {2, 2, 2, 6};
  int split_dim_ = 3;
  int num_split_ = 2;  // len of split_sizes_
  std::vector<int> split_sizes_{6, 6};
  size_t input_size, output1_size, output2_size;
  std::string inputPpath = "./test_data/splitfp32_input.bin";
  std::string output1Ppath = "./test_data/splitfp32_output1.bin";
  std::string output2Ppath = "./test_data/splitfp32_output2.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(inputPpath.c_str(), &input_size));
  auto output_data1 = reinterpret_cast<float *>(mindspore::lite::ReadFile(output1Ppath.c_str(), &output1_size));
  auto output_data2 = reinterpret_cast<float *>(mindspore::lite::ReadFile(output2Ppath.c_str(), &output2_size));
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(split_dim_, num_split_, split_sizes_);
    TestMain({{input_shape, input_data, VAR}}, {{output_shape1, output_data1}, {output_shape2, output_data2}}, param,
             fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestOpenCL_Split, input3_axis0) {
  std::vector<int> input_shape = {8, 1, 1, 1};
  std::vector<int> output_shape1 = {2, 1, 1, 1};
  std::vector<int> output_shape2 = {3, 1, 1, 1};
  std::vector<int> output_shape3 = {3, 1, 1, 1};
  int split_dim_ = 0;
  int num_split_ = 3;  // len of split_sizes_
  std::vector<int> split_sizes_{2, 3, 3};
  float input_data[] = {0.75, 0.06, 0.74, 0.30, 0.9, 0.59, 0.03, 0.37};
  float output_data1[] = {0.75, 0.06};
  float output_data2[] = {0.74, 0.30, 0.9};
  float output_data3[] = {0.59, 0.03, 0.37};
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(split_dim_, num_split_, split_sizes_);
    TestMain({{input_shape, input_data, VAR}},
             {{output_shape1, output_data1}, {output_shape2, output_data2}, {output_shape3, output_data3}}, param,
             fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}
}  // namespace mindspore::lite::opencl::test
