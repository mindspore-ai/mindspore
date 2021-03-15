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
#include "nnacl/conv_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Conv2dTranspose : public CommonTest {};

namespace {
// PrimitiveType_DeConv2D: src/ops/populate/deconv2d_populate.cc
OpParameter *CreateParameter(int n, int h, int w, int ci, int co, int kh, int kw, std::vector<int> pad, int oh, int ow,
                             std::vector<int> *input_shape, std::vector<int> *weight_shape,
                             std::vector<int> *bias_shape, std::vector<int> *output_shape) {
  auto *param = test::CreateParameter<ConvParameter>(schema::PrimitiveType_Conv2dTransposeFusion);
  param->kernel_h_ = kh;
  param->kernel_w_ = kw;
  param->stride_h_ = 2;
  param->stride_w_ = 2;
  MS_ASSERT(pad.size() == 4);
  param->pad_u_ = pad[0];
  param->pad_d_ = pad[1];
  param->pad_l_ = pad[2];
  param->pad_r_ = pad[3];
  param->dilation_h_ = 1;
  param->dilation_w_ = 1;
  param->act_type_ = ActType_No;
  param->group_ = 1;

  *input_shape = {n, h, w, ci};
  *weight_shape = {ci, kh, kw, co};
  *bias_shape = {co};
  *output_shape = {1, oh, ow, co};
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Conv2dTranspose, test0) {
  int n = 1;
  int h = 2;
  int w = 2;
  int oh = 4;
  int ow = 4;
  int ci = 2;
  int co = 1;
  int kh = 2;
  int kw = 2;
  std::vector<int> pad = {0, 0, 0, 0};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {0, 2, 4, 6, 1, 3, 5, 7};
  float bias_data[] = {0.5};
  float output_data[] = {1.5, 3.5, 3.5, 13.5, 5.5, 7.5, 23.5, 33.5, 5.5, 23.5, 7.5, 33.5, 41.5, 59.5, 59.5, 85.5};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param =
      CreateParameter(n, h, w, ci, co, kh, kw, pad, oh, ow, &input_shape, &weight_shape, &bias_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Conv2dTranspose, test1) {
  int n = 1;
  int h = 3;
  int w = 3;
  int oh = 5;
  int ow = 5;
  int ci = 2;
  int co = 1;
  int kh = 2;
  int kw = 2;
  std::vector<int> pad = {0, 0, 0, 0};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  float weight_data[] = {0, 2, 4, 6, 1, 3, 5, 7};
  float bias_data[] = {0.5};
  float output_data[] = {1.5,  3.5,  3.5,  13.5, 5.5,  5.5,   7.5,  23.5, 33.5, 41.5, 7.5,  33.5, 9.5,
                         43.5, 11.5, 59.5, 85.5, 77.5, 111.5, 95.5, 13.5, 63.5, 15.5, 73.5, 17.5};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param =
      CreateParameter(n, h, w, ci, co, kh, kw, pad, oh, ow, &input_shape, &weight_shape, &bias_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}
TEST_F(TestOpenCL_Conv2dTranspose, test2) {
  int n = 1;
  int h = 2;
  int w = 2;
  int oh = 5;
  int ow = 5;
  int ci = 2;
  int co = 1;
  int kh = 3;
  int kw = 3;
  std::vector<int> pad = {0, 0, 0, 0};
  float input_data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  float weight_data[] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                         1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0};
  float bias_data[] = {0.5};
  float output_data[] = {1.5,   3.5,   8.5,  13.5, 23.5,  7.5,   9.5,   44.5,  43.5,  53.5,  18.5,  38.5, 128.5,
                         106.5, 142.5, 59.5, 77.5, 180.5, 111.5, 137.5, 113.5, 131.5, 312.5, 189.5, 215.5};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param =
      CreateParameter(n, h, w, ci, co, kh, kw, pad, oh, ow, &input_shape, &weight_shape, &bias_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Conv2dTranspose, test0MultiBatch) {
  int n = 2;
  int h = 2;
  int w = 2;
  int oh = 4;
  int ow = 4;
  int ci = 2;
  int co = 1;
  int kh = 2;
  int kw = 2;
  std::vector<int> pad = {0, 0, 0, 0};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {0, 2, 4, 6, 1, 3, 5, 7};
  float bias_data[] = {0.5};
  float output_data[] = {1.5, 3.5, 3.5, 13.5, 5.5, 7.5, 23.5, 33.5, 5.5, 23.5, 7.5, 33.5, 41.5, 59.5, 59.5, 85.5,
                         1.5, 3.5, 3.5, 13.5, 5.5, 7.5, 23.5, 33.5, 5.5, 23.5, 7.5, 33.5, 41.5, 59.5, 59.5, 85.5};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param =
      CreateParameter(n, h, w, ci, co, kh, kw, pad, oh, ow, &input_shape, &weight_shape, &bias_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_Conv2dTranspose, test1MultiBatch) {
  int n = 2;
  int h = 3;
  int w = 3;
  int oh = 5;
  int ow = 5;
  int ci = 2;
  int co = 1;
  int kh = 2;
  int kw = 2;
  std::vector<int> pad = {0, 0, 0, 0};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
  float weight_data[] = {0, 2, 4, 6, 1, 3, 5, 7};
  float bias_data[] = {0.5};
  float output_data[] = {1.5,  3.5,  3.5,  13.5, 5.5,   5.5,   7.5,  23.5, 33.5, 41.5, 7.5,  33.5, 9.5,
                         43.5, 11.5, 59.5, 85.5, 77.5,  111.5, 95.5, 13.5, 63.5, 15.5, 73.5, 17.5, 1.5,
                         3.5,  3.5,  13.5, 5.5,  5.5,   7.5,   23.5, 33.5, 41.5, 7.5,  33.5, 9.5,  43.5,
                         11.5, 59.5, 85.5, 77.5, 111.5, 95.5,  13.5, 63.5, 15.5, 73.5, 17.5};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param =
      CreateParameter(n, h, w, ci, co, kh, kw, pad, oh, ow, &input_shape, &weight_shape, &bias_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}
TEST_F(TestOpenCL_Conv2dTranspose, test2MultiBatch) {
  int n = 2;
  int h = 2;
  int w = 2;
  int oh = 5;
  int ow = 5;
  int ci = 2;
  int co = 1;
  int kh = 3;
  int kw = 3;
  std::vector<int> pad = {0, 0, 0, 0};
  float input_data[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  float weight_data[] = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                         1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0};
  float bias_data[] = {0.5};
  float output_data[] = {1.5,   3.5,   8.5,  13.5,  23.5,  7.5,   9.5,   44.5,  43.5,  53.5,  18.5,  38.5,  128.5,
                         106.5, 142.5, 59.5, 77.5,  180.5, 111.5, 137.5, 113.5, 131.5, 312.5, 189.5, 215.5, 1.5,
                         3.5,   8.5,   13.5, 23.5,  7.5,   9.5,   44.5,  43.5,  53.5,  18.5,  38.5,  128.5, 106.5,
                         142.5, 59.5,  77.5, 180.5, 111.5, 137.5, 113.5, 131.5, 312.5, 189.5, 215.5};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param =
      CreateParameter(n, h, w, ci, co, kh, kw, pad, oh, ow, &input_shape, &weight_shape, &bias_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}
}  // namespace mindspore::lite::opencl::test
