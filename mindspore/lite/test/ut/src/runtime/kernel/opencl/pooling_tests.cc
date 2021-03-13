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
#include "nnacl/pooling_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Pooling : public CommonTest {};

namespace {
// PrimitiveType_Pooling: src/ops/populate/pooling_populate.cc
OpParameter *CreateParameter(PoolMode pool_mode, int window_h, int window_w, int stride_h, int stride_w, int pad_u,
                             int pad_d, int pad_l, int pad_r, RoundMode round_mode = RoundMode_Floor,
                             ActType act_type = ActType_No) {
  auto *param = test::CreateParameter<PoolingParameter>(schema::PrimitiveType_MaxPoolFusion);
  param->global_ = false;
  param->window_w_ = window_w;
  param->window_h_ = window_h;
  param->pad_u_ = pad_u;
  param->pad_d_ = pad_d;
  param->pad_l_ = pad_l;
  param->pad_r_ = pad_r;
  param->stride_w_ = stride_w;
  param->stride_h_ = stride_h;
  param->avg_mode_ = 0;
  param->pool_mode_ = pool_mode;
  param->round_mode_ = round_mode;
  param->act_type_ = act_type;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Pooling, Avg) {
  std::vector<int> input_shape = {1, 2, 2, 4};
  std::vector<int> output_shape = {1, 1, 1, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float output_data[] = {6, 7, 8, 9};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(PoolMode_AvgPool, 2, 2, 2, 2, 0, 0, 0, 0);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Pooling, Max) {
  std::vector<int> input_shape = {1, 2, 2, 4};
  std::vector<int> output_shape = {1, 1, 1, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float output_data[] = {12, 13, 14, 15};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(PoolMode_MaxPool, 2, 2, 2, 2, 0, 0, 0, 0);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Pooling, AvgMultiBatch) {
  std::vector<int> input_shape = {2, 2, 2, 4};
  std::vector<int> output_shape = {2, 1, 1, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float output_data[] = {6, 7, 8, 9, 6, 7, 8, 9};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(PoolMode_AvgPool, 2, 2, 2, 2, 0, 0, 0, 0);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Pooling, MaxMultiBatch) {
  std::vector<int> input_shape = {2, 2, 2, 4};
  std::vector<int> output_shape = {2, 1, 1, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  float output_data[] = {12, 13, 14, 15, 12, 13, 14, 15};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(PoolMode_MaxPool, 2, 2, 2, 2, 0, 0, 0, 0);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
