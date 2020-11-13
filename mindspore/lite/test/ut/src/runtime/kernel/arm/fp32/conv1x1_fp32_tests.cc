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
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "nnacl/matmul_parameter.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1_fp32.h"

namespace mindspore {
using mindspore::lite::Tensor;

class TestConv1x1Fp32 : public mindspore::CommonTest {
 public:
  TestConv1x1Fp32() = default;
};

TEST_F(TestConv1x1Fp32, Input1x1PrePack1) {
  auto conv_param = new ConvParameter();
  float in[] = {-0.59, -0.63,  -7.26,  -0.64,  -6.403, 4.87,   9.612,  9.36,   12.84,  -0.838, 6.588,  2.02,   13.756,
                15.92, 16.0,   -7.82,  9.53,   1.77,   10.521, 13.45,  17.991, 17.063, 4.6859, 13.57,  -6.31,  5.27,
                7.54,  -7.418, 15.12,  0.6195, 1.5475, -5.925, -7.59,  18.13,  15.8,   19.86,  -7.766, 13.25,  7.141,
                -0.34, 16.254, -5.78,  16.13,  -7.1,   6.259,  10.771, -5.54,  10.477, 9.2366, 12.258, -9.86,  -8.29,
                -4.9,  18.14,  -5.400, 0.829,  7.4575, 12.075, 13.734, 16.51,  -9.82,  -4.9,   18.44,  -0.808, 8.066,
                6.914, 2.5098, 10.985, 16.96,  1.721,  -1.0,   2.096,  9.2553, 8.635,  9.2136, 13.558, 7.7505, -0.55,
                15.68, -7.3,   0.429,  -0.560, 17.98,  19.068, 9.2764, 17.939, -6.51,  -2.04,  7.29,   -0.87,  10.311,
                -6.74, -6.424, 18.708, -0.368, 9.725,  9.129,  6.99,   3.11,   -1.573, -8.25,  10.427, 17.427, -9.739,
                17.32, 6.076,  -3.5,   7.43,   -2.659, -0.89,  -9.157, 1.9951, -3.463, 15.22,  13.99,  4.39,   18.12};
  float correct[] = {0.000, 0.000, 0.000,  0.000, 0.000, 0.000, 15.12, -7.59, -7.766, 0.000,
                     0.000, 0.429, 9.2764, 7.29,  0.000, 0.000, 0.000, 0.000, 0.000,  0.000};

  conv_param->input_h_ = 9;
  conv_param->input_w_ = 13;
  conv_param->input_channel_ = 1;
  conv_param->output_h_ = 4;
  conv_param->output_w_ = 5;
  conv_param->stride_h_ = conv_param->stride_w_ = 4;
  conv_param->pad_u_ = conv_param->pad_l_ = 2;

  float out[20] = {0};
  Conv1x1InputPack(in, out, conv_param, sizeof(float));
  EXPECT_EQ(0, CompareOutputData(out, correct, 20));
  delete conv_param;
}

TEST_F(TestConv1x1Fp32, Input1x1PrePack2) {
  auto conv_param = new ConvParameter();
  float in[] = {
    12.755477,  7.647509,   14.670943, -8.03628,   -1.815172, 7.7517915,  5.6838546,  0.9693578,  10.86119,  10.960915,
    17.758,     -4.800611,  -8.743361, 1.6797531,  -0.234721, 7.7575417,  10.19116,   11.744166,  -2.674233, 8.977257,
    1.5364298,  14.600166,  16.625568, -4.820712,  10.050005, 4.114301,   10.436717,  -7.443196,  -2.669484, 5.3399734,
    7.5060234,  12.705402,  -2.203446, 19.582493,  8.716431,  11.463841,  2.1704009,  -7.740846,  0.6420606, 15.4524,
    1.9975507,  -4.6742086, -0.425350, 7.120687,   -9.663703, 18.799034,  -4.425679,  10.846515,  -1.993019, 0.2714671,
    -8.511215,  16.797249,  18.438688, 8.391737,   15.632475, 16.98368,   -5.901906,  -2.718238,  -3.131561, -3.707477,
    -8.04332,   13.010143,  3.187699,  7.6656003,  9.344805,  2.100789,   -7.123898,  10.088698,  7.8578715, -8.320831,
    6.821173,   -2.263130,  -2.886815, 2.285673,   10.664816, -4.747543,  -4.9607406, 1.0546302,  15.628643, 1.7381196,
    18.267065,  11.504781,  -0.193673, 16.431538,  8.011203,  -3.3506372, 16.546675,  -3.983052,  4.8116174, -9.49816,
    11.714877,  12.401133,  -3.799531, 5.109032,   11.657709, 1.9226302,  0.9720376,  14.517606,  7.712793,  17.820406,
    17.644344,  15.314725,  17.884249, -3.6718662, -2.053803, 10.629432,  16.67133,   -3.929358,  3.3747706, 8.818307,
    -0.371532,  18.14205,   5.9272094, 12.691162,  6.816437,  8.310599,   17.566565,  16.581955,  -7.433713, 2.5550082,
    9.1433325,  -2.9258926, 5.7442937, -2.9434314, -9.864248, -0.122141,  11.5717945, -4.174809,  -6.192147, 8.390994,
    -7.4617224, 17.419308,  7.0560303, 11.58972,   17.671894, 6.2352304,  13.778206,  3.4766717,  -6.687946, -7.887233,
    -1.150991,  -3.1441534, 17.288366, 13.669407,  -4.997481, -6.147624,  -5.6006193, -8.15764,   9.595266,  8.296087,
    -0.9590447, -3.6464965, -8.155689, 4.8459644,  19.75259,  5.5307946,  -6.934994,  -9.928046,  4.02548,   -9.45412,
    13.605555,  10.22008,   -3.067481, 8.114803,   2.4563003, 0.4125615,  6.076172,   -1.875376,  19.553644, -9.809106,
    17.235031,  -4.222316,  -9.534478, 18.639902,  1.7095382, 18.821035,  -8.177748,  -2.9353676, 2.064462,  12.190292,
    -1.475221,  -1.842325,  -3.664825, 10.538533,  -4.255415, 3.4860964,  11.418711,  -2.348281,  -4.527373, 19.534836};
  float correct[] = {12.755477, -8.03628,  5.6838546, 10.960915,  7.5060234,  19.582493, 2.1704009,
                     15.4524,   -8.04332,  7.6656003, -7.123898,  -8.320831,  11.714877, 5.109032,
                     0.9720376, 17.820406, 9.1433325, -2.9434314, 11.5717945, 8.390994,  -0.9590447,
                     4.8459644, -6.934994, -9.45412,  -1.4752215, 10.538533,  11.418711, 19.534836};

  conv_param->input_h_ = 19;
  conv_param->input_w_ = 10;
  conv_param->input_channel_ = 1;
  conv_param->output_h_ = 7;
  conv_param->output_w_ = 4;
  conv_param->stride_h_ = conv_param->stride_w_ = 3;
  conv_param->pad_u_ = conv_param->pad_l_ = 0;

  float out[28] = {0};
  Conv1x1InputPack(in, out, conv_param, sizeof(float));
  ASSERT_EQ(0, CompareOutputData(out, correct, 28, 0.0001));
  delete conv_param;
}

TEST_F(TestConv1x1Fp32, Input1x1PrePack3) {
  auto conv_param = new ConvParameter();
  conv_param->input_channel_ = 2;
  conv_param->input_h_ = conv_param->input_w_ = 3;
  conv_param->output_h_ = conv_param->output_w_ = 3;
  conv_param->stride_h_ = conv_param->stride_w_ = 2;
  conv_param->pad_u_ = conv_param->pad_l_ = 1;

  float in[] = {1.6767339, 12.25904,  19.018835, 3.0790641,  -9.252135, -8.685675, 3.6115494, 3.2282279, 17.025112,
                -5.052577, 12.750252, 12.701241, -8.9477215, -9.080522, 19.03931,  -6.501229, -4.122992, 9.540845};
  float out[18] = {0};
  float correct[] = {0.0,       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.025112,
                     -5.052577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  Conv1x1InputPack(in, out, conv_param, sizeof(float));
  EXPECT_EQ(0, CompareOutputData(out, correct, 18));
  delete conv_param;
}

TEST_F(TestConv1x1Fp32, Input1x1PrePack4) {
  auto conv_param = new ConvParameter();
  conv_param->input_channel_ = 6;
  conv_param->input_h_ = conv_param->input_w_ = 3;
  conv_param->output_h_ = conv_param->output_w_ = 3;
  conv_param->stride_h_ = conv_param->stride_w_ = 2;
  conv_param->pad_u_ = conv_param->pad_l_ = 1;
  float in[] = {4.1795, 13.142, -3.593, 16.505, 19.899, 8.5562, 19.969, -6.235, -2.380, -9.027, 9.5542,
                18.974, 23.622, 8.3608, 47.325, -14.36, 15.370, 4.3049, -0.784, 37.925, -0.081, 6.1298,
                0.6721, -1.517, 37.998, 13.719, 11.029, 1.7127, -1.770, 41.903, 9.0560, 14.988, 3.1866,
                0.0562, 8.1381, 9.1391, 14.530, -14.10, -8.115, -8.071, -8.158, 7.7566, 19.250, 17.923,
                13.584, 3.3293, 9.7341, 18.834, -1.514, -0.293, 18.686, 0.0873, 4.2010, -2.253};
  float correct[] = {0.0,    0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    0.0,    0.0,    0.0,
                     0.0,    0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 37.998, 13.719, 11.029, 1.7127,
                     -1.770, 41.903, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    0.0,    0.0,    0.0,
                     0.0,    0.0,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    0.0};
  float out[54] = {0};
  Conv1x1InputPack(in, out, conv_param, sizeof(float));
  EXPECT_EQ(0, CompareOutputData(out, correct, 54));
  delete conv_param;
}

TEST_F(TestConv1x1Fp32, Conv1x1WeightTest1) {
  auto *conv_param = new ConvParameter();
  float in[] = {0.214637,  0.3815,   0.811557, 0.982146,  0.09123,   0.687198,  0.02742,   0.3360,    0.853275,
                0.674123,  0.81337,  0.57188,  0.706416,  0.2740942, 0.9045,    0.07155,   0.130864,  0.037712,
                0.5369175, 0.97283,  0.92133,  0.3588165, 0.7432479, 0.7886823, 0.870324,  0.230946,  0.343969,
                0.095415,  0.50036,  0.396918, 0.09029,   0.934583,  0.91616,   0.206713,  0.9756054, 0.614025,
                0.432057,  0.1493,   0.6787,   0.10642,   0.736823,  0.377668,  0.2464896, 0.93152,   0.315917,
                0.35745,   0.52233,  0.0263,   0.339392,  0.99447,   0.49129,   0.675686,  0.75703,   0.6665356,
                0.0491,    0.1070,   0.18899,  0.929156,  0.4633427, 0.08585,   0.040709,  0.2478724, 0.5238441,
                0.0579918, 0.531636, 0.085524, 0.640923,  0.336395,  0.218651,  0.630491};
  float co[] = {0.214637,  0.81337,   0.92133,   0.09029,  0.3815,    0.57188,   0.3588165, 0.934583,  0.811557,
                0.706416,  0.7432479, 0.91616,   0.982146, 0.2740942, 0.7886823, 0.206713,  0.09123,   0.9045,
                0.870324,  0.9756054, 0.687198,  0.07155,  0.230946,  0.614025,  0.02742,   0.130864,  0.343969,
                0.432057,  0.3360,    0.037712,  0.095415, 0.1493,    0.853275,  0.5369175, 0.50036,   0.6787,
                0.674123,  0.97283,   0.396918,  0.10642,  0,         0,         0,         0,         0,
                0,         0,         0,         0.736823, 0.49129,   0.040709,  0,         0.377668,  0.675686,
                0.2478724, 0,         0.2464896, 0.75703,  0.5238441, 0,         0.93152,   0.6665356, 0.0579918,
                0,         0.315917,  0.0491,    0.531636, 0,         0.35745,   0.1070,    0.085524,  0,
                0.52233,   0.18899,   0.640923,  0,        0.0263,    0.929156,  0.336395,  0,         0.339392,
                0.4633427, 0.218651,  0,         0.99447,  0.08585,   0.630491,  0,         0,         0,
                0,         0,         0,         0,        0,         0};

  conv_param->input_channel_ = 10;
  conv_param->output_channel_ = 7;
  float out[96] = {0};
  Pack1x1WeightFp32(in, out, conv_param);
  EXPECT_EQ(0, CompareOutputData(out, co, 96));
  delete conv_param;
}

int Conv1x1TestInit1(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                     ConvParameter *conv_param, float **correct) {
  auto *in_t = new lite::Tensor(kNumberTypeFloat, {1, 2, 3, 4}, schema::Format_NHWC, lite::Tensor::VAR);
  in_t->MallocData();
  float in[] = {12.216284, 3.3466918,  15.327419, 5.234958,  0.804376,   9.952188,  14.727955,  -8.080715,
                13.71383,  8.055829,   6.5845337, -9.25232,  -4.24519,   11.550042, 9.262012,   1.2780352,
                6.7263746, -3.9301445, 3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  memcpy(in_t->MutableData(), in, sizeof(float) * 24);
  inputs_->push_back(in_t);

  auto *weight_t = new lite::Tensor(kNumberTypeFloat, {3, 1, 1, 4}, schema::Format_NHWC, lite::Tensor::CONST_TENSOR);
  weight_t->MallocData();
  float weight[] = {-0.7308652, 0.5257509,  -0.87825793, -1.123181,   -1.2206168, 0.562695,
                    1.5382664,  -0.5020635, 0.8591602,   -0.26410004, 1.1262615,  0.073132955}; /* nhwc */
  memcpy(weight_t->MutableData(), weight, sizeof(float) * 12);
  inputs_->push_back(weight_t);

  auto *bias_t = new lite::Tensor(kNumberTypeFloat, {3}, schema::Format_NHWC, lite::Tensor::CONST_TENSOR);
  bias_t->MallocData();
  float bias[] = {2, 2, 2};
  memcpy(bias_t->MutableData(), bias, sizeof(float) * 3);
  inputs_->push_back(bias_t);

  auto *out_t = new lite::Tensor(kNumberTypeFloat, {1, 2, 3, 3}, schema::Format_NHWC, lite::Tensor::VAR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  *correct = reinterpret_cast<float *>(malloc(out_t->ElementsNum() * sizeof(float)));
  float co[] = {2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.3731456, 1.6877825, 12.427691, 2., 2., 2.};
  memcpy(*correct, co, out_t->ElementsNum() * sizeof(float));

  conv_param->kernel_h_ = conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = conv_param->stride_w_ = 2;
  conv_param->dilation_h_ = conv_param->dilation_w_ = 1;
  conv_param->pad_u_ = conv_param->pad_l_ = 1;
  conv_param->act_type_ = ActType_No;
  return out_t->ElementsNum();
}

TEST_F(TestConv1x1Fp32, Conv1x1Test1) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto conv_param = new ConvParameter();
  auto *ctx = new lite::InnerContext();
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  float *correct;
  int total_size = Conv1x1TestInit1(&inputs_, &outputs_, conv_param, &correct);
  auto *conv1x1 =
    new kernel::Convolution1x1CPUKernel(reinterpret_cast<OpParameter *>(conv_param), inputs_, outputs_, ctx, nullptr);

  conv1x1->Init();
  conv1x1->Run();

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete conv_param;
  delete conv1x1;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}

int Conv1x1TestInit2(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_,
                     ConvParameter *conv_param, float **correct) {
  size_t buffer_size;
  auto *in_t = new lite::Tensor(kNumberTypeFloat, {1, 300, 300, 24}, schema::Format_NHWC, lite::Tensor::VAR);
  in_t->MallocData();
  std::string input_path = "./conv/conv1x1fp32_input1_nhwc.bin";
  auto in = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &buffer_size));
  memcpy(in_t->MutableData(), in, buffer_size);
  inputs_->push_back(in_t);

  auto *weight_t = new lite::Tensor(kNumberTypeFloat, {40, 1, 1, 24}, schema::Format_NHWC, lite::Tensor::CONST_TENSOR);
  weight_t->MallocData();
  std::string weight_path = "./conv/conv1x1fp32_weight1_nhwc.bin";
  auto weight = reinterpret_cast<float *>(mindspore::lite::ReadFile(weight_path.c_str(), &buffer_size));
  memcpy(weight_t->MutableData(), weight, buffer_size);
  inputs_->push_back(weight_t);

  auto *bias_t = new lite::Tensor(kNumberTypeFloat, {40}, schema::Format_NHWC, lite::Tensor::CONST_TENSOR);
  bias_t->MallocData();
  std::string bias_path = "./conv/conv1x1fp32_bias1_nhwc.bin";
  auto bias = mindspore::lite::ReadFile(bias_path.c_str(), &buffer_size);
  memcpy(bias_t->MutableData(), bias, buffer_size);
  inputs_->push_back(bias_t);

  auto *out_t = new lite::Tensor(kNumberTypeFloat, {1, 300, 300, 40}, schema::Format_NHWC, lite::Tensor::VAR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  std::string out_path = "./conv/conv1x1fp32_output1_nhwc.bin";
  auto out_nhwc = mindspore::lite::ReadFile(out_path.c_str(), &buffer_size);
  *correct = reinterpret_cast<float *>(malloc(buffer_size));
  memcpy(*correct, out_nhwc, buffer_size);

  conv_param->kernel_h_ = conv_param->kernel_w_ = 1;
  conv_param->stride_h_ = conv_param->stride_w_ = 1;
  conv_param->dilation_h_ = conv_param->dilation_w_ = 1;
  conv_param->pad_u_ = conv_param->pad_l_ = 0;
  conv_param->act_type_ = ActType_No;
  return out_t->ElementsNum();
}

TEST_F(TestConv1x1Fp32, Conv1x1Test2) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto conv_param = new ConvParameter();
  auto *ctx = new lite::InnerContext();
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  float *correct;
  int total_size = Conv1x1TestInit2(&inputs_, &outputs_, conv_param, &correct);
  auto *conv1x1 =
    new kernel::Convolution1x1CPUKernel(reinterpret_cast<OpParameter *>(conv_param), inputs_, outputs_, ctx, nullptr);

  conv1x1->Init();
  conv1x1->Run();
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));

  /* running warm up */
  for (int i = 0; i < 0; i++) {
    conv1x1->Run();
  }

  /* running time cost */
  int loop_count = 1;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    conv1x1->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  uint64_t time_avg = cost / loop_count;
  printf("1x1 average time : %f ms\n", time_avg / 1000.0f);

  delete conv_param;
  delete conv1x1;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
  free(correct);
}
}  // namespace mindspore
