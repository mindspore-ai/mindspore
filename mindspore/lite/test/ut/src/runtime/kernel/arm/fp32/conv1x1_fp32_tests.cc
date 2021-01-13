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
}  // namespace mindspore
