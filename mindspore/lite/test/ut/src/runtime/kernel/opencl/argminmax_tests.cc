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
#include "nnacl/arg_min_max_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_ArgMinMax : public CommonTest {};

namespace {
// PrimitiveType_ArgMinFusion: src/ops/populate/argmin_populate.cc
// PrimitiveType_ArgMaxFusion: src/ops/populate/argmax_populate.cc
OpParameter *CreateParameter(schema::PrimitiveType type, int axis, int topk, bool out_value, bool keep_dims = false,
                             int axis_type = 0) {
  auto *param = test::CreateParameter<ArgMinMaxParameter>(type);
  param->axis_ = axis;
  param->topk_ = topk;
  param->axis_type_ = axis_type;
  param->out_value_ = out_value;
  param->keep_dims_ = keep_dims;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_ArgMinMax, axis0topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 0;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {3, 2, 2, 2};
  std::vector<int> output_shape = {2, 2, 2, 2};
  float input_data[] = {100, 2, 4, 50, 11, 12, 34, 35, 10, 20, 40, 5, 7, 80, 10, 11, 55, 25, 5, 15, 18, 8, 15, 16};
  float output_data[] = {0, 2, 1, 0, 2, 1, 0, 0, 2, 1, 2, 2, 0, 0, 2, 2};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis0topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 0;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {3, 2, 2, 2};
  std::vector<int> output_shape = {2, 2, 2, 2};
  float input_data[] = {100, 2, 4, 50, 11, 12, 34, 35, 10, 20, 40, 5, 7, 80, 10, 11, 55, 25, 5, 15, 18, 8, 15, 16};
  float output_data[] = {100, 25, 40, 50, 18, 80, 34, 35, 55, 20, 5, 15, 11, 12, 15, 16};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis1topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 1;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {2, 3, 2, 3};
  std::vector<int> output_shape = {2, 2, 2, 3};
  float input_data[] = {100, 2,  200, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                        10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  float output_data[] = {0, 1, 0, 1, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 0, 2, 1, 0, 0, 0, 1, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis1topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 1;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {2, 3, 2, 3};
  std::vector<int> output_shape = {2, 2, 2, 3};
  float input_data[] = {100, 2,  200, 4,  50, 6,  11, 12, 13, 34, 35, 36,  9,  6, 17, 10, 20, 30,
                        10,  20, 30,  40, 5,  60, 7,  80, 90, 10, 11, 120, 18, 5, 16, 9,  22, 23};
  float output_data[] = {100, 12, 200, 34, 50, 36,  11, 6,  17, 10, 35, 30,
                         18,  80, 90,  40, 22, 120, 10, 20, 30, 10, 11, 60};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 2;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {2, 3, 3, 3};
  std::vector<int> output_shape = {2, 3, 1, 3};
  float input_data[] = {10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12,
                        10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12,
                        10, 20, 30, 11, 15, 10, 5, 10, 12, 10, 20, 30, 11, 15, 10, 5, 10, 12};
  float output_data[] = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis2topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 2;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 2, 5};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {30, 45, 30, 50, 90, 20, 20, 25, 40, 50, 30, 45, 30, 50, 90, 20, 20, 25, 40, 50,
                         30, 45, 30, 50, 90, 20, 20, 25, 40, 50, 30, 45, 30, 50, 90, 20, 20, 25, 40, 50};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis2topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 2;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 2, 5};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {2, 2, 0, 2, 0, 1, 0, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 0, 1,
                         2, 2, 0, 2, 0, 1, 0, 2, 0, 1, 2, 2, 0, 2, 0, 1, 0, 2, 0, 1};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis3topk2index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 3;
  int topk = 2;
  bool out_value = false;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 3, 2};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1, 4, 3, 4, 0, 3, 1};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_ArgMinMax, axis3topk2value) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 3;
  int topk = 2;
  bool out_value = true;
  std::vector<int> input_shape = {2, 2, 3, 5};
  std::vector<int> output_shape = {2, 2, 3, 2};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90,
                        20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50,
                        30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30};
  float output_data[] = {90, 40, 50, 20, 50, 45, 90, 40, 50, 20, 50, 45,
                         90, 40, 50, 20, 50, 45, 90, 40, 50, 20, 50, 45};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim32axis1topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 1;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {1, 2, 14};
  std::vector<int> output_shape = {1, 14};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50,
                        30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25};
  float output_data[] = {1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim43axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 2;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {2, 2, 2, 14};
  std::vector<int> output_shape = {2, 2, 14};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15,
                        1,  50, 30, 45, 25, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50, 30, 10, 20, 30,
                        40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25,
                        50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 10, 20, 30, 40, 90, 20, 11, 15,
                        1,  50, 30, 45, 25, 50, 30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25};
  float output_data[] = {1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
                         1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim21axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 0;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {2, 14};
  std::vector<int> output_shape = {14};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25, 50,
                        30, 10, 20, 30, 40, 90, 20, 11, 15, 1,  50, 30, 45, 25};
  float output_data[] = {1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim10axis2topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 0;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {14};
  std::vector<int> output_shape = {1};
  float input_data[] = {10, 20, 30, 40, 90, 20, 11, 15, 1, 50, 30, 45, 25, 50};
  float output_data[] = {4};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim43axis1topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 1;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {2, 2, 2, 14};
  std::vector<int> output_shape = {1, 2, 2, 14};
  float input_data[] = {
    2732.,  10799., 9845.,  3264.,  13123., 4859.,  14019., 15719., 9225.,  7891.,  4373.,  5874.,  14116., 14935.,
    15430., 15832., 6744.,  3468.,  14650., 705.,   15846., 2599.,  10327., 2222.,  7768.,  2897.,  9893.,  537.,
    11085., 6216.,  6921.,  6036.,  2163.,  5072.,  4851.,  7877.,  2046.,  1871.,  7599.,  2496.,  15186., 8291.,
    10200., 15537., 755.,   797.,   659.,   3219.,  15246., 8615.,  7456.,  16321., 3337.,  2745.,  4735.,  8736.,
    6687.,  714.,   2292.,  8343.,  10915., 14846., 11723., 11122., 1207.,  6172.,  8994.,  10368., 10368., 10148.,
    7221.,  6021.,  3622.,  3560.,  8948.,  12561., 14671., 12676., 1641.,  11306., 13754., 14879., 4984.,  4353.,
    13633., 12263., 12201., 10297., 14627., 12134., 11383., 15115., 8622.,  7250.,  4187.,  14208., 10638., 2659.,
    9781.,  2956.,  10873., 16298., 12372., 2251.,  4420.,  13062., 7108.,  1071.,  12927., 14324., 5251.,  13260.};
  float output_data[] = {1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1,
                         1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
TEST_F(TestOpenCL_ArgMinMax, dim43axis3topk1index) {
  schema::PrimitiveType type = schema::PrimitiveType_ArgMaxFusion;
  int axis = 3;
  int topk = 1;
  bool out_value = false;
  std::vector<int> input_shape = {1, 13, 13, 6};
  std::vector<int> output_shape = {1, 13, 13};
  float input_data[] = {
    2732.,  10799., 9845.,  3264.,  13123., 4859.,  14019., 15719., 9225.,  7891.,  4373.,  5874.,  14116., 14935.,
    15430., 15832., 6744.,  3468.,  14650., 705.,   15846., 2599.,  10327., 2222.,  7768.,  2897.,  9893.,  537.,
    11085., 6216.,  6921.,  6036.,  2163.,  5072.,  4851.,  7877.,  2046.,  1871.,  7599.,  2496.,  15186., 8291.,
    10200., 15537., 755.,   797.,   659.,   3219.,  15246., 8615.,  7456.,  16321., 3337.,  2745.,  4735.,  8736.,
    6687.,  714.,   2292.,  8343.,  10915., 14846., 11723., 11122., 1207.,  6172.,  8994.,  10368., 10368., 10148.,
    7221.,  6021.,  3622.,  3560.,  8948.,  12561., 14671., 12676., 1641.,  11306., 13754., 14879., 4984.,  4353.,
    13633., 12263., 12201., 10297., 14627., 12134., 11383., 15115., 8622.,  7250.,  4187.,  14208., 10638., 2659.,
    9781.,  2956.,  10873., 16298., 12372., 2251.,  4420.,  13062., 7108.,  1071.,  12927., 14324., 5251.,  13260.,
    7012.,  9396.,  14312., 3918.,  9359.,  1684.,  11491., 7098.,  15127., 10959., 2957.,  4469.,  14165., 8752.,
    13617., 9797.,  14505., 5795.,  1472.,  7263.,  7365.,  11870., 8448.,  6001.,  3762.,  13604., 10146., 9008.,
    16221., 2435.,  1634.,  15914., 973.,   4464.,  10215., 11157., 8393.,  10623., 13824., 14218., 2418.,  12843.,
    13242., 3455.,  6167.,  5819.,  12418., 6521.,  6242.,  7742.,  9123.,  15070., 14459., 10179., 6738.,  14254.,
    2787.,  7316.,  4305.,  2610.,  5531.,  6926.,  15401., 15418., 15041., 7204.,  6922.,  4182.,  15403., 13160.,
    10251., 15106., 307.,   13392., 14368., 5302.,  1152.,  6950.,  8467.,  5294.,  13866., 13683., 1208.,  2492.,
    10728., 15949., 14622., 13592., 8829.,  770.,   15875., 8286.,  11490., 5995.,  15629., 14192., 2344.,  13640.,
    3091.,  12895., 3912.,  1434.,  6594.,  5368.,  8372.,  10563., 7148.,  7997.,  3854.,  8032.,  15620., 8131.,
    4845.,  10379., 5116.,  10838., 3533.,  2937.,  9837.,  4939.,  15032., 9744.,  3224.,  5021.,  10389., 1134.,
    25.,    9680.,  956.,   1913.,  2934.,  13429., 9661.,  13907., 2721.,  10088., 928.,   15588., 5627.,  11003.,
    6265.,  5446.,  469.,   10527., 8717.,  1863.,  1720.,  5272.,  591.,   6185.,  2322.,  15912., 11702., 207.,
    15115., 4262.,  14447., 3421.,  12281., 5249.,  15071., 10102., 11052., 8408.,  15741., 8216.,  12355., 11218.,
    5103.,  7939.,  2282.,  1740.,  6118.,  12579., 5846.,  13310., 15037., 3781.,  2775.,  2603.,  14368., 7179.,
    13928., 6356.,  1162.,  14006., 12267., 13733., 15997., 16028., 623.,   15848., 8962.,  13851., 4051.,  1241.,
    10903., 9013.,  4403.,  1198.,  13460., 2997.,  5661.,  15939., 10787., 807.,   11657., 2121.,  12585., 15255.,
    8067.,  3886.,  8922.,  6066.,  15212., 9987.,  1823.,  15113., 11658., 11803., 12717., 199.,   1447.,  5181.,
    10581., 13153., 11308., 11042., 10146., 5208.,  6177.,  14981., 14568., 4863.,  6180.,  1792.,  1483.,  14626.,
    8389.,  894.,   14261., 5374.,  15440., 13758., 136.,   10429., 6273.,  10449., 9584.,  14627., 13688., 3419.,
    168.,   6004.,  2852.,  12464., 9753.,  4419.,  8039.,  8700.,  15139., 3186.,  5918.,  5149.,  1777.,  3361.,
    8338.,  5393.,  4317.,  14676., 4605.,  2562.,  6213.,  13669., 9100.,  4652.,  13173., 11005., 13634., 11887.,
    6235.,  11605., 423.,   11815., 16331., 11926., 11678., 11921., 6854.,  967.,   4370.,  9052.,  6187.,  5203.,
    433.,   13097., 6237.,  13486., 1429.,  12489., 12121., 2546.,  14816., 14299., 329.,   3612.,  16363., 8401.,
    6761.,  14266., 3968.,  8150.,  15935., 1040.,  6250.,  8356.,  8798.,  7704.,  6772.,  5311.,  9411.,  9523.,
    12424., 9144.,  13147., 11357., 6011.,  2798.,  13399., 8352.,  2195.,  4680.,  6599.,  9303.,  3085.,  15674.,
    5713.,  5240.,  10100., 11191., 15168., 10955., 732.,   5028.,  8473.,  13088., 7594.,  12046., 4566.,  9500.,
    7444.,  16338., 3396.,  13846., 5347.,  7034.,  595.,   647.,   12232., 573.,   6797.,  5637.,  8448.,  11400.,
    11471., 14799., 16309., 5259.,  9220.,  6567.,  4444.,  2989.,  13594., 586.,   14132., 5102.,  7601.,  11483.,
    11059., 739.,   13161., 4882.,  11637., 5410.,  15923., 16030., 437.,   3898.,  12203., 1847.,  9724.,  1020.,
    6930.,  941.,   11095., 8641.,  11590., 5610.,  11317., 9008.,  15454., 2107.,  14672., 9882.,  13948., 4259.,
    11834., 945.,   13418., 8393.,  7468.,  1805.,  15225., 1862.,  8742.,  3751.,  9864.,  15373., 2040.,  903.,
    14032., 15352., 14870., 8696.,  8015.,  14297., 5896.,  12003., 7942.,  7377.,  9671.,  14804., 5593.,  16322.,
    13884., 12688., 3128.,  7026.,  3821.,  2711.,  8472.,  1028.,  2660.,  13292., 2353.,  10583., 5662.,  7734.,
    8345.,  12052., 7521.,  10597., 10937., 12695., 15771., 1053.,  2977.,  5491.,  3893.,  2679.,  11187., 4950.,
    14838., 12295., 2665.,  3057.,  14473., 6838.,  3968.,  851.,   9592.,  5028.,  3793.,  7316.,  8053.,  7152.,
    3331.,  8318.,  5930.,  8769.,  5652.,  804.,   5444.,  3024.,  112.,   1967.,  650.,   4333.,  1384.,  13278.,
    14171., 13867., 63.,    3999.,  3988.,  2502.,  13577., 3516.,  12891., 2671.,  13731., 2387.,  10060., 5394.,
    3441.,  8010.,  10466., 13537., 1963.,  5763.,  2956.,  7396.,  3898.,  3969.,  14705., 7296.,  4903.,  13336.,
    8890.,  292.,   14691., 9029.,  14470., 4099.,  5346.,  7033.,  4776.,  14780., 13729., 7452.,  6980.,  4122.,
    736.,   10488., 4461.,  1971.,  11465., 13749., 8389.,  13217., 1671.,  10877., 606.,   2120.,  12534., 6996.,
    9351.,  1731.,  10453., 15835., 7788.,  3395.,  6246.,  8020.,  10567., 8787.,  5343.,  2304.,  11909., 3419.,
    1131.,  15262., 14281., 2003.,  11783., 11413., 10213., 7644.,  13704., 1707.,  9774.,  8192.,  7528.,  691.,
    13862., 13401., 11338., 2547.,  10978., 2683.,  8535.,  15456., 6995.,  12570., 6862.,  6176.,  11379., 6598.,
    5985.,  4524.,  827.,   10041., 6834.,  10413., 14057., 3204.,  11705., 93.,    15707., 13713., 2467.,  3778.,
    404.,   5037.,  9401.,  13263., 375.,   16036., 3945.,  10942., 15876., 497.,   7666.,  7373.,  9630.,  13677.,
    14167., 8930.,  4515.,  6729.,  3290.,  10167., 1562.,  13686., 13334., 8652.,  15055., 15714., 11866., 3123.,
    15334., 1838.,  16080., 15933., 9660.,  6959.,  14330., 14440., 4736.,  3466.,  4043.,  6029.,  12615., 4702.,
    5638.,  7853.,  15605., 5534.,  13839., 14505., 6310.,  13621., 2987.,  4690.,  11655., 3292.,  2881.,  5801.,
    15170., 7282.,  11100., 8526.,  8933.,  9435.,  15606., 8292.,  2463.,  10461., 13490., 7676.,  8366.,  8797.,
    7794.,  3745.,  4876.,  3808.,  9961.,  9040.,  9282.,  5576.,  13299., 2173.,  9354.,  4720.,  6874.,  1179.,
    8888.,  7288.,  12609., 2496.,  2757.,  12120., 7458.,  4047.,  2051.,  6844.,  3310.,  7845.,  15531., 1747.,
    11096., 15942., 7828.,  9094.,  3868.,  4723.,  4998.,  4930.,  604.,   8156.,  3686.,  9061.,  3451.,  3781.,
    13421., 9545.,  100.,   4790.,  9037.,  6037.,  5627.,  8863.,  3665.,  3107.,  8429.,  15603., 14586., 14728.,
    6910.,  9497.,  21.,    6573.,  1253.,  6102.,  8592.,  13465., 9198.,  3191.,  9893.,  8063.,  13697., 13701.,
    1734.,  6540.,  3418.,  8778.,  15355., 5046.,  7246.,  9022.,  9800.,  14535., 16173., 3205.,  15919., 12987.,
    5290.,  4547.,  6282.,  4850.,  1337.,  3547.,  13657., 387.,   5245.,  10958., 3922.,  1221.,  14010., 1924.,
    7185.,  8901.,  8639.,  350.,   8856.,  3715.,  12613., 8616.,  4260.,  7738.,  9393.,  3511.,  10904., 673.,
    1938.,  8033.,  12750., 8945.,  7303.,  1973.,  13035., 12334., 15856., 15348., 10879., 15265., 8529.,  5277.,
    11788., 11894., 10030., 11126., 12576., 7970.,  115.,   5719.,  10876., 12697., 11438., 10738., 8043.,  15924.,
    8169.,  12910., 5696.,  3404.,  12150., 10825., 4242.,  2820.,  1799.,  2691.,  9264.,  11340., 6437.,  12404.,
    9709.,  9776.,  6253.,  10194., 10419., 10801., 11335., 16218., 11697., 4078.,  5405.,  4611.,  8266.,  15956.,
    6634.,  11585., 6007.,  3604.,  3280.,  5162.,  5618.,  28.,    1434.,  2903.,  3252.,  6448.,  14274., 9830.,
    8969.,  7426.,  11636., 15212., 14057., 13145., 13692., 9077.,  612.,   4186.,  9284.,  8809.,  9738.,  4108.,
    5736.,  10465., 10661., 4263.,  9120.,  9594.,  11553., 9114.,  99.,    7385.,  2354.,  10584., 15570., 5908.,
    13022., 16028., 1608.,  5394.,  13721., 9112.,  7719.,  11498., 13947., 284.,   13976., 11922., 16067., 14508.,
    16083., 10541., 11062., 0.,     10378., 4803.};
  float output_data[] = {4, 1, 3, 2, 4, 5, 4, 1, 3, 1, 1, 1, 4, 3, 4, 1, 5, 3, 1, 0, 0, 2, 5, 2, 3, 1, 2, 1, 1,
                         1, 0, 0, 5, 4, 2, 1, 1, 0, 4, 2, 5, 3, 3, 5, 2, 2, 0, 5, 0, 3, 1, 2, 3, 3, 2, 2, 1, 1,
                         1, 0, 1, 1, 0, 3, 1, 0, 0, 5, 1, 4, 4, 2, 4, 2, 3, 2, 1, 1, 2, 4, 4, 0, 5, 2, 4, 2, 0,
                         2, 1, 0, 5, 0, 3, 3, 2, 4, 2, 0, 3, 0, 2, 2, 0, 1, 2, 2, 3, 3, 1, 2, 1, 4, 1, 2, 2, 3,
                         2, 4, 2, 5, 2, 2, 3, 1, 0, 4, 2, 1, 2, 2, 0, 2, 0, 2, 5, 3, 5, 4, 2, 3, 1, 1, 1, 0, 0,
                         4, 4, 1, 0, 4, 4, 1, 2, 5, 1, 5, 1, 3, 3, 0, 4, 3, 0, 4, 2, 5, 2, 4, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(type, axis, topk, out_value);
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable, 1e-1, 1e-1, true);
  }
}
}  // namespace mindspore::lite::opencl::test
