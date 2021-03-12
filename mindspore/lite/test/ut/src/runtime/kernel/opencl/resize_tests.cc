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
#include "nnacl/resize_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Resize : public CommonTest {};

namespace {
// PrimitiveType_Resize: src/ops/populate/resize_populate.cc
OpParameter *CreateParameter(schema::ResizeMethod method, int new_height, int new_width, bool align_corners) {
  auto *param = test::CreateParameter<ResizeParameter>(schema::PrimitiveType_Resize);
  param->new_height_ = new_height;
  param->new_width_ = new_width;
  if (align_corners) {
    param->coordinate_transform_mode_ = 1;
  }
  param->method_ = method;
  param->preserve_aspect_ratio_ = false;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Resize, Bilinear) {
  schema::ResizeMethod method = schema::ResizeMethod_LINEAR;
  int oh = 4;
  int ow = 4;
  bool align_corners = false;

  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, oh, ow, 1};
  float input_data[] = {0, 1, 2, 3};
  float output_data[] = {0, 0.5, 1, 1, 1, 1.5, 2, 2, 2, 2.5, 3, 3, 2, 2.5, 3, 3};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(method, oh, ow, align_corners);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Resize, Bilinear_AlignCorners) {
  schema::ResizeMethod method = schema::ResizeMethod_LINEAR;
  int oh = 3;
  int ow = 3;
  bool align_corners = true;

  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, oh, ow, 1};
  float input_data[] = {0, 1, 2, 3};
  float output_data[] = {0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(method, oh, ow, align_corners);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Resize, NEAREST) {
  schema::ResizeMethod method = schema::ResizeMethod_NEAREST;
  int oh = 4;
  int ow = 4;
  bool align_corners = false;

  std::vector<int> input_shape = {1, 2, 2, 1};
  std::vector<int> output_shape = {1, oh, ow, 1};
  float input_data[] = {0, 1, 2, 3};
  float output_data[] = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(method, oh, ow, align_corners);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Resize, BilinearBatch) {
  schema::ResizeMethod method = schema::ResizeMethod_LINEAR;
  int oh = 4;
  int ow = 4;
  bool align_corners = false;

  std::vector<int> input_shape = {2, 2, 2, 1};
  std::vector<int> output_shape = {2, oh, ow, 1};
  float input_data[] = {0, 1, 2, 3, 0, 1, 2, 3};
  float output_data[] = {0, 0.5, 1, 1, 1, 1.5, 2, 2, 2, 2.5, 3, 3, 2, 2.5, 3, 3,
                         0, 0.5, 1, 1, 1, 1.5, 2, 2, 2, 2.5, 3, 3, 2, 2.5, 3, 3};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(method, oh, ow, align_corners);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Resize, Bilinear_AlignCornersBatch) {
  schema::ResizeMethod method = schema::ResizeMethod_LINEAR;
  int oh = 3;
  int ow = 3;
  bool align_corners = true;

  std::vector<int> input_shape = {2, 2, 2, 1};
  std::vector<int> output_shape = {2, oh, ow, 1};
  float input_data[] = {0, 1, 2, 3, 0, 1, 2, 3};
  float output_data[] = {0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3, 0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(method, oh, ow, align_corners);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Resize, NEARESTBatch) {
  schema::ResizeMethod method = schema::ResizeMethod_NEAREST;
  int oh = 4;
  int ow = 4;
  bool align_corners = false;

  std::vector<int> input_shape = {2, 2, 2, 1};
  std::vector<int> output_shape = {2, oh, ow, 1};
  float input_data[] = {0, 1, 2, 3, 0, 1, 2, 3};
  float output_data[] = {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3,
                         0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(method, oh, ow, align_corners);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
