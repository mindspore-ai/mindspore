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

class TestOpenCL_DepthwiseConv2d : public CommonTest {};

namespace {
// PrimitiveType_DepthwiseConv2D: src/ops/populate/depthwise_conv2d_populate.cc
OpParameter *CreateParameter(int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_u, int pad_d, int pad_l,
                             int pad_r, int dilation_h, int dilation_w, ActType act_type, int input_channel) {
  auto *param = test::CreateParameter<ConvParameter>(schema::PrimitiveType_DepthwiseConv2D);
  param->kernel_h_ = kernel_h;
  param->kernel_w_ = kernel_w;
  param->stride_h_ = stride_h;
  param->stride_w_ = stride_w;
  param->pad_u_ = pad_u;
  param->pad_d_ = pad_d;
  param->pad_l_ = pad_l;
  param->pad_r_ = pad_r;
  param->input_channel_ = input_channel;
  param->dilation_h_ = dilation_h;
  param->dilation_w_ = dilation_w;
  param->act_type_ = act_type;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_DepthwiseConv2d, NoPad) {
  int kernel_h = 3;
  int kernel_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_u = 0;
  int pad_d = 0;
  int pad_l = 0;
  int pad_r = 0;
  int dilation_h = 1;
  int dilation_w = 1;
  ActType act_type = ActType_No;

  std::vector<int> input_shape = {1, 4, 4, 4};
  std::vector<int> output_shape = {1, 2, 2, 4};
  std::vector<int> weight_shape = {1, kernel_h, kernel_w, output_shape.back()};
  std::vector<int> bias_shape = {output_shape.back()};
  float input_data[] = {0.5488135,  0.0202184,  0.45615032, 0.31542835, 0.71518934, 0.83261985, 0.56843394, 0.36371076,
                        0.60276335, 0.77815676, 0.0187898,  0.57019675, 0.5448832,  0.87001216, 0.6176355,  0.43860152,
                        0.4236548,  0.9786183,  0.6120957,  0.9883738,  0.6458941,  0.7991586,  0.616934,   0.10204481,
                        0.4375872,  0.46147937, 0.94374806, 0.20887676, 0.891773,   0.7805292,  0.6818203,  0.16130951,
                        0.96366274, 0.11827443, 0.3595079,  0.6531083,  0.3834415,  0.639921,   0.43703195, 0.2532916,
                        0.79172504, 0.14335328, 0.6976312,  0.46631077, 0.5288949,  0.9446689,  0.06022547, 0.2444256,
                        0.56804454, 0.5218483,  0.6667667,  0.15896958, 0.92559665, 0.41466194, 0.67063785, 0.11037514,
                        0.07103606, 0.2645556,  0.21038257, 0.6563296,  0.0871293,  0.7742337,  0.12892629, 0.13818295};
  float bias_data[] = {0, 0, 0, 0};
  float weight_data[] = {0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,
                         0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772,
                         0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051,
                         0.5759465,  0.9292962,  0.31856894, 0.6674104,  0.13179787, 0.7163272,  0.2894061,  0.18319136,
                         0.5865129,  0.02010755, 0.82894003, 0.00469548};
  float output_data[] = {3.3848767, 1.4446403, 1.8428744, 1.3194335, 2.5873442, 2.1384869, 2.04022,  1.1872686,
                         2.2294958, 1.6570128, 2.465089,  1.4294086, 2.7941442, 1.7871612, 2.188921, 1.0601988};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(kernel_h, kernel_w, stride_h, stride_w, pad_u, pad_d, pad_l, pad_r, dilation_h,
                                  dilation_w, act_type, input_shape.back());
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-2 : 1e-5);
  }
}

TEST_F(TestOpenCL_DepthwiseConv2d, Pad) {
  int kernel_h = 3;
  int kernel_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_u = 1;
  int pad_d = 1;
  int pad_l = 1;
  int pad_r = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  ActType act_type = ActType_No;

  std::vector<int> input_shape = {1, 3, 3, 5};
  std::vector<int> output_shape = {1, 3, 3, 5};
  std::vector<int> weight_shape = {1, kernel_h, kernel_w, output_shape.back()};
  std::vector<int> bias_shape = {output_shape.back()};
  float input_data[] = {0.5488135,  0.3834415,  0.77815676, 0.9446689, 0.6120957,  0.71518934, 0.79172504, 0.87001216,
                        0.5218483,  0.616934,   0.60276335, 0.5288949, 0.9786183,  0.41466194, 0.94374806, 0.5448832,
                        0.56804454, 0.7991586,  0.2645556,  0.6818203, 0.4236548,  0.92559665, 0.46147937, 0.7742337,
                        0.3595079,  0.6458941,  0.07103606, 0.7805292, 0.45615032, 0.43703195, 0.4375872,  0.0871293,
                        0.11827443, 0.56843394, 0.6976312,  0.891773,  0.0202184,  0.639921,   0.0187898,  0.06022547,
                        0.96366274, 0.83261985, 0.14335328, 0.6176355, 0.6667667};
  float weight_data[] = {0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,
                         0.10204481, 0.20887676, 0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958,
                         0.11037514, 0.6563296,  0.13818295, 0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,
                         0.09609841, 0.97645944, 0.4686512,  0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696,
                         0.12019656, 0.2961402,  0.11872772, 0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146,
                         0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962};
  float bias_data[] = {0, 0, 0, 0, 0};
  float output_data[] = {1.189188,   1.0425153,  1.8012011,  0.6074867,  1.2120346,  1.5005531,  0.8346756, 2.4365785,
                         0.54975945, 1.6815965,  1.2690231,  0.60214907, 1.6158017,  0.42115876, 0.8854959, 1.1709145,
                         1.0929465,  1.3534508,  1.1985044,  1.2932993,  2.4621446,  1.7086457,  2.6977584, 2.1960166,
                         2.3769147,  2.3185873,  0.6133741,  0.9687358,  0.9987654,  1.0254729,  0.8368954, 0.74171704,
                         0.8749627,  0.8953936,  0.5093431,  1.5496738,  0.54936385, 0.7683113,  1.165742,  1.3682933,
                         1.0517888,  0.59817517, 0.75649744, 1.2075498,  0.38804203};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(kernel_h, kernel_w, stride_h, stride_w, pad_u, pad_d, pad_l, pad_r, dilation_h,
                                  dilation_w, act_type, input_shape.back());
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, weight_data, CONST_TENSOR},
              {bias_shape, bias_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-2 : 1e-5);
  }
}

}  // namespace mindspore::lite::opencl::test
