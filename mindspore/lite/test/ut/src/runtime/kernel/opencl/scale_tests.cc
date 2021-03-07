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
#include "nnacl/scale.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Scale : public CommonTest {};

namespace {
// PrimitiveType_Resize: src/ops/populate/scale_populate.cc
OpParameter *CreateParameter(int axis, int activation_type = schema::ActivationType_NO_ACTIVATION) {
  auto *param = test::CreateParameter<ScaleParameter>(schema::PrimitiveType_ScaleFusion);
  param->axis_ = axis;
  param->activation_type_ = activation_type;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Scale, Axis1) {
  int axis = 1;
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> weight_shape = {input_shape[axis]};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float scale_data[] = {1, 2};
  float offset_data[] = {1, 2};
  float output_data[] = {2, 3, 4, 5, 6, 7, 16, 18, 20, 22, 24, 26};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, scale_data, CONST_TENSOR},
              {weight_shape, offset_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Scale, Axis3) {
  int axis = 3;
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> weight_shape = {input_shape[axis]};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float scale_data[] = {1, 2, 3};
  float offset_data[] = {1, 2, 3};
  float output_data[] = {2, 6, 12, 5, 12, 21, 8, 18, 30, 11, 24, 39};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, scale_data, CONST_TENSOR},
              {weight_shape, offset_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Scale, Axis3RELU6) {
  int axis = 3;
  std::vector<int> input_shape = {1, 2, 2, 3};
  std::vector<int> weight_shape = {input_shape[axis]};
  std::vector<int> output_shape = input_shape;
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float scale_data[] = {1, 2, -1};
  float offset_data[] = {1, 2, 3};
  float output_data[] = {2, 6, 0, 5, 6, 0, 6, 6, 0, 6, 6, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis, schema::ActivationType_RELU6);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, scale_data, CONST_TENSOR},
              {weight_shape, offset_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
