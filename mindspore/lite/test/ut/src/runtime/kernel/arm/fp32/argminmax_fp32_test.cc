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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/fp32/arg_min_max.h"
#include "mindspore/lite/nnacl/arg_min_max.h"
#include "mindspore/lite/nnacl/arithmetic_common.h"

namespace mindspore {

class TestArgMinMaxTestFp32 : public mindspore::CommonTest {
 public:
  TestArgMinMaxTestFp32() = default;
};

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest1) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {2, 2, 0, 2, 0};
  std::vector<int> shape = {3, 5};
  float out[5];
  ArgMinMaxParameter param;
  param.topk_ = 1;
  param.out_value_ = false;
  param.axis_ = 0;
  param.data_type_ = 43;
  param.dims_size_ = 2;
  param.get_max_ = true;
  param.keep_dims_ = false;
  ArgMinMax(in.data(), out, shape.data(), &param);
  for (size_t i = 0; i < except_out.size(); ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.000001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest1_keep_dim) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {2, 2, 0, 2, 0};
  std::vector<int> shape = {3, 5};
  float out[5];
  ArgMinMaxParameter param;
  param.topk_ = 1;
  param.out_value_ = false;
  param.axis_ = 0;
  param.data_type_ = 43;
  param.dims_size_ = 2;
  param.get_max_ = true;
  param.keep_dims_ = true;
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(shape[param.axis_] * sizeof(ArgElement)));
  std::vector<int> out_shape = {1, 5};
  ComputeStrides(shape.data(), param.in_strides_, shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  ArgMinMax(in.data(), out, shape.data(), &param);
  for (size_t i = 0; i < except_out.size(); ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.000001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest_axis2_keep_dim) {
  std::vector<float> in = {10, 20, 30, 11, 15, 10, 5,  10, 12, 10, 20, 30, 11, 15,
                           10, 5,  10, 12, 10, 20, 30, 11, 15, 10, 5,  10, 12};
  std::vector<float> except_out = {1, 0, 0, 1, 0, 0, 1, 0, 0};
  std::vector<int> shape = {1, 3, 3, 3};
  float out[9];
  ArgMinMaxParameter param;
  param.topk_ = 1;
  param.out_value_ = false;
  param.axis_ = 2;
  param.data_type_ = 43;
  param.dims_size_ = 4;
  param.get_max_ = true;
  param.keep_dims_ = true;
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(shape[param.axis_] * sizeof(ArgElement)));
  std::vector<int> out_shape = {1, 3, 1, 3};
  ComputeStrides(shape.data(), param.in_strides_, shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  ArgMinMax(in.data(), out, shape.data(), &param);
  for (size_t i = 0; i < except_out.size(); ++i) {
    std::cout << out[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.000001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest2) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {30, 45, 30, 50, 90};
  std::vector<int> shape = {3, 5};
  float out[5];
  ArgMinMaxParameter param;
  param.topk_ = 1;
  param.out_value_ = true;
  param.axis_ = 0;
  param.data_type_ = 43;
  param.dims_size_ = 2;
  param.get_max_ = true;
  param.keep_dims_ = false;
  ArgMinMax(in.data(), out, shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.000001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMinTest2) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {10, 11, 15, 1, 30};
  std::vector<int> shape = {3, 5};
  float out[5];
  ArgMinMaxParameter param;
  param.topk_ = 1;
  param.out_value_ = true;
  param.axis_ = 0;
  param.data_type_ = 43;
  param.dims_size_ = 2;
  param.get_max_ = false;
  param.keep_dims_ = false;
  ArgMinMax(in.data(), out, shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.000001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest3_axis2_out_data) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {30, 45, 30, 50, 90, 20, 20, 25, 40, 50};
  ArgMinMaxParameter param;
  param.axis_ = 2;
  std::vector<int> in_shape = {1, 1, 3, 5};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = true;
  param.topk_ = 2;
  std::vector<int> out_shape = {1, 1, 2, 5};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[10];
  ArgMaxDim2(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest3_axis2_out_index) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {2, 2, 0, 2, 0, 1, 0, 2, 0, 1};
  ArgMinMaxParameter param;
  param.axis_ = 2;
  std::vector<int> in_shape = {1, 1, 3, 5};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = false;
  param.topk_ = 2;
  std::vector<int> out_shape = {1, 1, 2, 5};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[10];
  ArgMaxDim2(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest4_axis3_out_data) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {90, 40, 50, 20, 50, 45};
  ArgMinMaxParameter param;
  param.axis_ = 3;
  std::vector<int> in_shape = {1, 1, 3, 5};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = true;
  param.topk_ = 2;
  std::vector<int> out_shape = {1, 1, 3, 2};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[6];
  ArgMaxDim3(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest4_axis3_out_index) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {4, 3, 4, 0, 3, 1};
  ArgMinMaxParameter param;
  param.axis_ = 3;
  std::vector<int> in_shape = {1, 1, 3, 5};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = false;
  param.topk_ = 2;
  std::vector<int> out_shape = {1, 1, 3, 2};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[6];
  ArgMaxDim3(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest5_axis1_out_index) {
  std::vector<float> in = {100, 2,  300, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                           10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  std::vector<float> except_out = {0, 1, 0, 1, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 0, 2, 1, 0, 0, 0, 1, 1, 0};
  ArgMinMaxParameter param;
  param.axis_ = 1;
  std::vector<int> in_shape = {2, 3, 2, 3};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = false;
  param.topk_ = 2;
  std::vector<int> out_shape = {2, 2, 2, 3};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[24];
  ArgMaxDim1(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest5_axis1_out_data) {
  std::vector<float> in = {100, 2,  300, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                           10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  std::vector<float> except_out = {100, 12, 300, 34, 50, 36,  11, 6,  17, 10, 35, 30,
                                   18,  80, 90,  40, 22, 120, 10, 20, 30, 10, 11, 60};
  ArgMinMaxParameter param;
  param.axis_ = 1;
  std::vector<int> in_shape = {2, 3, 2, 3};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = true;
  param.topk_ = 2;
  std::vector<int> out_shape = {2, 2, 2, 3};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[24];
  ArgMaxDim1(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest6_axis0_out_index) {
  std::vector<float> in = {100, 2, 4, 50, 11, 12, 34, 35, 10, 20, 40, 5, 7, 80, 10, 11, 55, 25, 5, 15, 18, 8, 15, 16};
  std::vector<float> except_out = {0, 2, 1, 0, 2, 1, 0, 0, 2, 1, 2, 2, 0, 0, 2, 2};
  ArgMinMaxParameter param;
  param.axis_ = 1;
  std::vector<int> in_shape = {3, 2, 2, 2};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = false;
  param.topk_ = 2;
  std::vector<int> out_shape = {2, 2, 2, 2};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[16];
  ArgMaxDim0(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMaxTest6_axis0_out_data) {
  std::vector<float> in = {100, 2, 4, 50, 11, 12, 34, 35, 10, 20, 40, 5, 7, 80, 10, 11, 55, 25, 5, 15, 18, 8, 15, 16};
  std::vector<float> except_out = {100, 25, 40, 50, 18, 80, 34, 35, 55, 20, 5, 15, 11, 12, 15, 16};
  ArgMinMaxParameter param;
  param.axis_ = 1;
  std::vector<int> in_shape = {3, 2, 2, 2};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = true;
  param.topk_ = 2;
  std::vector<int> out_shape = {2, 2, 2, 2};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[16];
  ArgMaxDim0(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

TEST_F(TestArgMinMaxTestFp32, ArgMinTest1_axis3_out_data) {
  std::vector<float> in = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50, 30};
  std::vector<float> except_out = {10, 20, 1, 11, 25, 30};
  ArgMinMaxParameter param;
  param.axis_ = 3;
  std::vector<int> in_shape = {1, 1, 3, 5};
  param.arg_elements_ = reinterpret_cast<ArgElement *>(malloc(in_shape[param.axis_] * sizeof(ArgElement)));
  param.out_value_ = true;
  param.topk_ = 2;
  std::vector<int> out_shape = {1, 1, 3, 2};
  ComputeStrides(in_shape.data(), param.in_strides_, in_shape.size());
  ComputeStrides(out_shape.data(), param.out_strides_, out_shape.size());
  float out[6];
  ArgMinDim3(in.data(), out, in_shape.data(), &param);
  ASSERT_EQ(0, CompareOutputData(out, except_out.data(), except_out.size(), 0.00001));
}

}  // namespace mindspore
