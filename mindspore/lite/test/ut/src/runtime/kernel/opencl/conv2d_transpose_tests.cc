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
OpParameter *CreateParameter(int n, int h, int w, int ci, int co, int kh, int kw, int pad,
                             std::vector<int> *input_shape, std::vector<int> *weight_shape,
                             std::vector<int> *bias_shape, std::vector<int> *output_shape) {
  auto *param = test::CreateParameter<ConvParameter>(schema::PrimitiveType_DeConv2D);
  param->kernel_h_ = kh;
  param->kernel_w_ = kw;
  param->stride_h_ = 2;
  param->stride_w_ = 2;
  param->pad_u_ = pad;
  param->pad_d_ = pad;
  param->pad_l_ = pad;
  param->pad_r_ = pad;
  param->dilation_h_ = 1;
  param->dilation_w_ = 1;
  param->act_type_ = ActType_No;

  int oh = 2 * h - 1 + 2 * (kh - 1 - pad) - kh + 1;
  int ow = 2 * w - 1 + 2 * (kw - 1 - pad) - kw + 1;
  *input_shape = {n, h, w, ci};
  *weight_shape = {co, kh, kw, ci};
  *bias_shape = {co};
  *output_shape = {1, oh, ow, co};
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Conv2dTranspose, test0) {
  int n = 1;
  int h = 2;
  int w = 2;
  int ci = 2;
  int co = 1;
  int kh = 2;
  int kw = 2;
  int pad = 0;
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float bias_data[] = {0.5};
  float output_data[] = {5.5, 6.5, 17.5, 22.5, 7.5, 8.5, 27.5, 32.5, 29.5, 38.5, 41.5, 54.5, 47.5, 56.5, 67.5, 80.5};

  for (auto fp16_enable : {false, true}) {
    std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
    auto *param =
      CreateParameter(n, h, w, ci, co, kh, kw, pad, &input_shape, &weight_shape, &bias_shape, &output_shape);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
