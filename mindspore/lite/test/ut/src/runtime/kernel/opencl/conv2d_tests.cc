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

class TestOpenCL_Conv2D : public CommonTest {};

namespace {
// PrimitiveType_Conv2D: src/ops/populate/conv2d_populate.cc
ConvParameter *CreateParameter(const std::string &attr, ActType act_type) {
  auto *param = test::CreateParameter<ConvParameter>(schema::PrimitiveType_Conv2DFusion);
  param->act_type_ = act_type;
  sscanf(attr.c_str(),
         "inputNHWC_%dx%dx%dx%d_outputNHWC_%dx%dx%dx%d_kernelHW_%dx%d_strideHW_%dx%d_padTopBottomLeftRight_%dx%dx%dx%d_"
         "dilationHW_%dx%d",
         &param->input_batch_, &param->input_h_, &param->input_w_, &param->input_channel_, &param->output_batch_,
         &param->output_h_, &param->output_w_, &param->output_channel_, &param->kernel_h_, &param->kernel_w_,
         &param->stride_h_, &param->stride_w_, &param->pad_u_, &param->pad_d_, &param->pad_l_, &param->pad_r_,
         &param->dilation_h_, &param->dilation_w_);
  return param;
}
}  // namespace

void TestMain_Conv2D(const std::string &attr, float *input_data, float *weight_data, float *bias_data,
                     float *output_data, ActType act_type, bool fp16_enable, float atol = 1e-9) {
  auto *param = CreateParameter(attr, act_type);
  param->group_ = 1;  // group conv is not supported in this test
  std::vector<int> input_shape = {param->input_batch_, param->input_h_, param->input_w_, param->input_channel_};
  std::vector<int> weight_shape = {param->output_channel_, param->kernel_h_, param->kernel_w_, param->input_channel_};
  std::vector<int> bias_shape = {param->output_channel_};
  std::vector<int> output_shape = {param->output_batch_, param->output_h_, param->output_w_, param->output_channel_};
  std::vector<ArgsTuple> input_infos = {{input_shape, input_data, VAR}, {weight_shape, weight_data, CONST_TENSOR}};
  if (bias_data) {
    input_infos.emplace_back(bias_shape, bias_data, CONST_TENSOR);
  }
  TestMain(input_infos, {output_shape, output_data}, reinterpret_cast<OpParameter *>(param), fp16_enable, atol);
}

TEST_F(TestOpenCL_Conv2D, test0) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  std::vector<int> input_shape, weight_shape, bias_shape, output_shape;
  float input_data[] = {0, 1, 2, 3, 4, 5, -6, -7};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  float bias_data[] = {0, 0};

  float output_data[] = {1, 1, 5, 5, 9, 9, -13, -13};
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, true, 1e-6f);

  float output_data_relu[] = {1, 1, 5, 5, 9, 9, 0, 0};
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data_relu, ActType_Relu, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data_relu, ActType_Relu, true, 1e-6f);

  float output_data_relu6[] = {1, 1, 5, 5, 6, 6, 0, 0};
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data_relu6, ActType_Relu6, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data_relu6, ActType_Relu6, true, 1e-6f);
}

TEST_F(TestOpenCL_Conv2D, test0_no_bias) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  float output_data[] = {1, 1, 5, 5, 9, 9, 13, 13};
  TestMain_Conv2D(attr, input_data, weight_data, nullptr, output_data, ActType_No, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, nullptr, output_data, ActType_No, true, 1e-6f);
}

TEST_F(TestOpenCL_Conv2D, test1) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float bias_data[] = {0.5, -0.5};
  float output_data[] = {2.5, 3.5, 8.5, 17.5, 14.5, 31.5, 20.5, 45.5};
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, true, 1e-6f);
}

TEST_F(TestOpenCL_Conv2D, test2) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x1_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x1x0x1_dilationHW_1x1";
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  float bias_data[] = {0};
  float output_data[] = {28, 18, 22, 13};
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, true, 1e-6f);
}

TEST_F(TestOpenCL_Conv2D, test3) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x1x0x1_dilationHW_1x1";
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float bias_data[] = {0.5, -0.5};
  float output_data[] = {168.5, 391.5, 80.5, 223.5, 60.5, 235.5, 20.5, 123.5};
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, true, 1e-6f);
}

TEST_F(TestOpenCL_Conv2D, test3_batch2) {
  std::string attr =
    "inputNHWC_2x2x2x2_outputNHWC_2x2x2x2_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x1x0x1_dilationHW_1x1";
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
  float weight_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float bias_data[] = {0.5, -0.5};
  float output_data[] = {168.5, 391.5, 80.5, 223.5, 60.5, 235.5, 20.5, 123.5,
                         168.5, 391.5, 80.5, 223.5, 60.5, 235.5, 20.5, 123.5};
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, false, 1e-3f);
  TestMain_Conv2D(attr, input_data, weight_data, bias_data, output_data, ActType_No, true, 1e-6f);
}

TEST_F(TestOpenCL_Conv2D, test4) {
  std::vector<std::tuple<std::string, std::string, std::vector<float>, std::vector<float>, std::vector<float>,
                         std::vector<float>, ActType>>
    cases = {
      {"SimpleTestFloat32WithAnisotropicStrides",
       "inputNHWC_1x3x6x1_outputNHWC_1x2x2x1_kernelHW_2x2_strideHW_1x3_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4, 5, 4, 3, -3, -4, -5},
       {1, 2, 3, 4},
       {-1},
       {30, -24, 40, -34},
       ActType_No},
      {"SimpleTestFloat32",
       "inputNHWC_2x2x4x1_outputNHWC_2x1x2x3_kernelHW_2x2_strideHW_2x2_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4},
       {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1},
       {1, 2, 3},
       {18, 2, 5, 18, 2, 5, 17, 4, 3, 37, 4, 3},
       ActType_No},
      {"SimpleTestFloat32SingleThreaded",
       "inputNHWC_2x2x4x1_outputNHWC_2x1x2x3_kernelHW_2x2_strideHW_2x2_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4},
       {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1},
       {1, 2, 3},
       {18, 2, 5, 18, 2, 5, 17, 4, 3, 37, 4, 3},
       ActType_No},
      {"SimpleTestFloat32WithChannels",
       "inputNHWC_2x2x4x2_outputNHWC_2x1x2x3_kernelHW_2x2_strideHW_2x2_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,   1,   1, 1, 1,   1,   1, 1,
        0.5, 0.5, 1,   1,   1.5, 1.5, 2,   2,   0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2},
       {1, 1, 2, 2, 3, 3, 4, 4, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1},
       {1, 2, 3},
       {18, 2, 5, 18, 2, 5, 17, 4, 3, 37, 4, 3},
       ActType_No},
      {"InputAndweightSameWidthHeight",
       "inputNHWC_2x2x4x1_outputNHWC_2x1x1x1_kernelHW_2x4_strideHW_2x2_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4},
       {1, 2, 3, 4, -1, -1, 1, 1},
       {0},
       {10, 34},
       ActType_No},
      {"ActivationRelu6Test",
       "inputNHWC_2x2x4x1_outputNHWC_2x1x2x3_kernelHW_2x2_strideHW_2x2_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4},
       {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1},
       {1, 2, 3},
       {6, 2, 5, 6, 2, 5, 6, 4, 3, 6, 4, 3},
       ActType_Relu6},
      {"StrideTest",
       "inputNHWC_2x2x4x1_outputNHWC_2x1x3x3_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 4, 4},
       {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1},
       {1, 2, 3},
       {18, 2, 5, 22, 3, 6, 21, 1, 6, 17, 4, 3, 31, 5, 4, 40, 3, 4},
       ActType_No},
      {"PaddingTest",
       "inputNHWC_1x2x4x1_outputNHWC_1x2x4x3_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x1x0x1_dilationHW_1x1",
       {1, 1, 1, 1, 2, 2, 3, 2},
       {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1},
       {1, 2, 3},
       {18, 2, 5, 22, 3, 6, 21, 1, 6, 8, -1, 4, 7, 2, -1, 9, 3, -2, 8, 1, -2, 3, 0, 1},
       ActType_No},
      {"PointwiseFloat32",
       "inputNHWC_2x2x4x2_outputNHWC_2x2x4x1_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,   1,   1, 1, 1,   1,   1, 1,
        0.5, 0.5, 1,   1,   1.5, 1.5, 2,   2,   0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2},
       {1, 2},
       {0},
       {1.5, 1.5, 1.5, 1.5, 3, 3, 3, 3, 1.5, 3, 4.5, 6, 1.5, 3, 4.5, 6},
       ActType_No},
      {"SimpleTestFloat32WithAnisotropicStrides",
       "inputNHWC_1x3x6x1_outputNHWC_1x2x2x1_kernelHW_2x2_strideHW_1x3_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4, 5, 4, 3, -3, -4, -5},
       {1, 2, 3, 4},
       {-1},
       {30, -24, 40, -34},
       ActType_No},
      {"HandCalculatedFloat32",
       "inputNHWC_1x3x4x1_outputNHWC_1x3x4x1_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_1x1x1x1_dilationHW_1x1",
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
       {1, 4, 7, 2, 5, 8, 3, 6, 9},
       {0},
       {105, 150, 183, 95, 235, 312, 357, 178, 187, 234, 261, 121},
       ActType_No},
      {"HandCalculatedFloat32WithConstweight",
       "inputNHWC_1x3x4x1_outputNHWC_1x3x4x1_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_1x1x1x1_dilationHW_1x1",
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
       {1, 4, 7, 2, 5, 8, 3, 6, 9},
       {0},
       {105, 150, 183, 95, 235, 312, 357, 178, 187, 234, 261, 121},
       ActType_No},
      {"HandCalculatedWithBiasFloat32",
       "inputNHWC_1x3x4x1_outputNHWC_1x3x4x1_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_1x1x1x1_dilationHW_1x1",
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
       {1, 4, 7, 2, 5, 8, 3, 6, 9},
       {10},
       {115, 160, 193, 105, 245, 322, 367, 188, 197, 244, 271, 131},
       ActType_No},
      {"HandCalculatedWithReluFloat32",
       "inputNHWC_1x3x4x1_outputNHWC_1x3x4x1_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_1x1x1x1_dilationHW_1x1",
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
       {1, 4, 7, 2, 5, 8, 3, 6, 9},
       {-200},
       {0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0},
       ActType_Relu},
      {"HandCalculatedValidFloat32",
       "inputNHWC_1x3x4x1_outputNHWC_1x1x2x1_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
       {1, 4, 7, 2, 5, 8, 3, 6, 9},
       {0},
       {312, 357},
       ActType_No},
      {"SimpleTestFloatWithDilation",
       "inputNHWC_1x9x9x1_outputNHWC_1x3x3x1_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_3x3",
       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
       {1, 2, 3, 4, 5, 6, 7, 8, 9},
       {0},
       {5, 5, 5, 5, 5, 5, 5, 5, 5},
       ActType_No},
      {"SimpleTestQuantizedOutputMultiplierGreaterThan1",
       "inputNHWC_2x2x4x1_outputNHWC_2x1x2x3_kernelHW_2x2_strideHW_2x2_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1",
       {1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4},
       {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1},
       {1, 2, 3},
       {18, 2, 5, 18, 2, 5, 17, 4, 3, 37, 4, 3},
       ActType_No},
    };

  for (auto &case_ : cases) {
    auto &name = std::get<0>(case_);
    auto &attr = std::get<1>(case_);
    auto input_data = std::get<2>(case_).data();
    auto weight_data = std::get<3>(case_).data();
    auto bias_data = std::get<4>(case_).data();
    auto expect_data = std::get<5>(case_).data();
    auto act_type = std::get<6>(case_);
    std::cout << name << std::endl;
    TestMain_Conv2D(attr, input_data, weight_data, bias_data, expect_data, act_type, false);
    TestMain_Conv2D(attr, input_data, weight_data, bias_data, expect_data, act_type, true);
  }
}

}  // namespace mindspore::lite::opencl::test
