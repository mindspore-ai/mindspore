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
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/power.h"

// PrimitiveType_PowFusion: src/ops/populate/power_populate.cc

using mindspore::lite::Tensor;
using mindspore::schema::Format::Format_NHWC;
namespace mindspore::lite::opencl::test {
class TestPowerOpenCLCI : public CommonTest {
 public:
  TestPowerOpenCLCI() {}
};
// PrimitiveType_Concat: src/ops/populate/concat_populate.cc
OpParameter *CreateParameter(bool broadcast_, float shift_, float scale_, float power_ = 2) {
  auto *param = test::CreateParameter<PowerParameter>(schema::PrimitiveType_PowFusion);
  param->power_ = power_;
  param->broadcast_ = broadcast_;
  param->shift_ = shift_;
  param->scale_ = scale_;
  return reinterpret_cast<OpParameter *>(param);
}

TEST_F(TestPowerOpenCLCI, Int32CI) {
  std::vector<int> input0_shape = {1, 2, 8};
  std::vector<int> input1_shape = {1, 2, 8};
  std::vector<int> output_shape = {1, 2, 8};
  bool broadcast_ = false;
  float shift_ = 0;
  float scale_ = 1;
  float input0_data[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};
  float input1_data[] = {2, 2, 2, 1, 2, 2, 3, 3, 2, 2, 3, 0, 2, 2, 1, 2};
  float output_data[] = {4.0,   9.0,   16.0,   5.0, 36.0,  49.0,  512,  729,
                         100.0, 121.0, 1728.0, 1.0, 196.0, 225.0, 16.0, 289.0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(broadcast_, shift_, scale_);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, VAR}}, {output_shape, output_data}, param,
             fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

TEST_F(TestPowerOpenCLCI, Fp32CI) {
  std::vector<int> input0_shape = {2, 8};
  std::vector<int> input1_shape = {2, 8};
  std::vector<int> output_shape = {2, 8};
  bool broadcast_ = false;
  float shift_ = 0;
  float scale_ = 1;
  float input0_data[] = {0.78957046,  -0.99770847, 1.05838929,  1.60738329,  -1.66226552, -2.03170525,
                         -0.48257631, -0.94244638, 1.47462044,  -0.80247114, 0.12354778,  -0.36436107,
                         -2.41973013, -0.40221205, -0.26739485, 0.23298305};
  float input1_data[] = {3, 2, 2, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2};
  float output_data[] = {0.49223521, 0.99542219, 1.12018788, 1.60738329, 2.76312667, 4.1278262,  0.23287989, 0.88820518,
                         3.20657016, 0.64395994, 0.01526405, 0.13275899, 5.85509388, 0.16177453, 0.07150001, 0.0542811};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(broadcast_, shift_, scale_);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, VAR}}, {output_shape, output_data}, param,
             fp16_enable, fp16_enable ? 1e-2 : 1e-6);
  }
}

TEST_F(TestPowerOpenCLCI, Fp32UnAlign) {
  std::vector<int> input0_shape = {2, 7};
  std::vector<int> input1_shape = {2, 7};
  std::vector<int> output_shape = {2, 7};
  bool broadcast_ = false;
  float shift_ = 0;
  float scale_ = 1;
  float input0_data[] = {0.78957046, -0.99770847, 1.05838929, 1.60738329,  -1.66226552, -2.03170525, -0.48257631,
                         1.47462044, -0.80247114, 0.12354778, -0.36436107, -2.41973013, -0.40221205, -0.26739485};
  float input1_data[] = {3, 2, 2, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2};
  float output_data[] = {0.49223521, 0.99542219, 1.12018788, 1.60738329, 2.76312667, 4.1278262,  0.23287989,
                         3.20657016, 0.64395994, 0.01526405, 0.13275899, 5.85509388, 0.16177453, 0.07150001};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(broadcast_, shift_, scale_);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, VAR}}, {output_shape, output_data}, param,
             fp16_enable, fp16_enable ? 1e-2 : 1e-6);
  }
}

TEST_F(TestPowerOpenCLCI, broadcast) {
  std::vector<int> input0_shape = {1, 2, 8};
  std::vector<int> output_shape = {1, 2, 8};
  bool broadcast_ = true;
  float shift_ = 0;
  float scale_ = 1;
  float input0_data[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};
  float output_data[] = {4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64, 81, 100.0, 121.0, 144, 169, 196.0, 225.0, 256, 289.0};
  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(broadcast_, shift_, scale_);
    TestMain({{input0_shape, input0_data, VAR}}, {output_shape, output_data}, param, fp16_enable,
             fp16_enable ? 1e-3 : 1e-6);
  }
}

TEST_F(TestPowerOpenCLCI, Fp16CI) {
  std::vector<int> input0_shape = {2, 8};
  std::vector<int> input1_shape = {2, 8};
  std::vector<int> output_shape = {2, 8};
  bool broadcast_ = false;
  float shift_ = 0;
  float scale_ = 1;
  float input0_data[] = {0.1531, -0.8003, -0.1848, 0.3833, -1.469, 0.5586, -0.3223, -0.8887,
                         0.697,  -1.007,  -0.45,   -1.736, -0.462, -0.699, -0.596,  0.7466};
  float input1_data[] = {2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0};
  float output_data[] = {0.02344, -0.8003, -0.1848, 0.147,  2.156,  0.312, 0.1039, 0.7896,
                         0.4856,  1.014,   0.2025,  -1.736, 0.2134, 0.489, -0.596, 0.7466};
  for (auto fp16_enable : {true}) {
    auto *param = CreateParameter(broadcast_, shift_, scale_);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, VAR}}, {output_shape, output_data}, param,
             fp16_enable, fp16_enable ? 1e-3 : 1e-6);
  }
}

}  // namespace mindspore::lite::opencl::test
