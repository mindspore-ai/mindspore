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
#include "nnacl/slice_parameter.h"
#include "ut/src/runtime/kernel/opencl/common.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Slice : public CommonTest {};

namespace {
// PrimitiveType_Slice: src/ops/populate/slice_populate.cc
OpParameter *CreateParameter(const std::vector<int> &begin, const std::vector<int> &size) {
  auto *param = test::CreateParameter<SliceParameter>(schema::PrimitiveType_SliceFusion);
  param->param_length_ = begin.size();
  for (int i = 0; i < begin.size(); ++i) {
    param->begin_[i] = begin[i];
    param->size_[i] = size[i];
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Slice, 4D) {
  float input_data[] = {-0.45816937, 0.92391545,  -0.9135602, -1.4002057, 1.1080881,  0.40712625,  -0.28128958,
                        0.09470133,  0.19801073,  0.04927751, -1.2808367, 0.1470597,  0.03393711,  -0.33282498,
                        -1.0433807,  -1.3678077,  -0.6423931, 0.5584889,  0.28965706, 0.5343769,   0.75480366,
                        -1.9328151,  -0.48714373, 1.711132,   -1.8871949, -0.2987629, -0.14000037, -0.080552,
                        0.95056856,  -0.06886655, 0.5316237,  0.05787678};
  float output_data[] = {-0.9135602,  -1.4002057,  1.1080881,  0.40712625, -0.28128958, -1.2808367, 0.1470597,
                         0.03393711,  -0.33282498, -1.0433807, 0.28965706, 0.5343769,   0.75480366, -1.9328151,
                         -0.48714373, -0.14000037, -0.080552,  0.95056856, -0.06886655, 0.5316237};
  auto param = CreateParameter({0, 0, 0, 2}, {1, 2, 2, 5});
  TestMain({{{1, 2, 2, 8}, input_data, VAR}}, {{1, 2, 2, 5}, output_data}, param, false);
}

TEST_F(TestOpenCL_Slice, test0) {
  std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<float>,
                         std::vector<int>, std::vector<int>>>
    cases = {{"In1D", {4}, {2}, {1, 2, 3, 4}, {2, 3}, {1}, {2}},
             {"In2D", {2, 3}, {1, 2}, {1, 2, 3, 4, 5, 6}, {4, 5}, {1, 0}, {1, 2}},
             {"In3D",
              {2, 3, 2},
              {2, 3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {0, 0, 0},
              {2, 3, 2}},
             {"InputFloat", {4, 1, 1, 1}, {3, 1, 1, 1}, {1, 2, 3, 4}, {2, 3, 4}, {1, 0, 0, 0}, {3, 1, 1, 1}},
             {"IndexInt64", {4, 1, 1, 1}, {3, 1, 1, 1}, {1, 2, 3, 4}, {2, 3, 4}, {1, 0, 0, 0}, {3, 1, 1, 1}},
             {"InputInteger1",
              {3, 2, 3, 1},
              {1, 1, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3},
              {1, 0, 0, 0},
              {1, 1, 3, 1}},
             {"InputInteger2",
              {3, 2, 3, 1},
              {1, 2, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3, 4, 4, 4},
              {1, 0, 0, 0},
              {1, 2, 3, 1}},
             {"InputInteger3",
              {3, 2, 3, 1},
              {2, 1, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3, 5, 5, 5},
              {1, 0, 0, 0},
              {2, 1, 3, 1}},
             {"SizeMinus1",
              {3, 2, 3, 1},
              {2, 1, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3, 5, 5, 5},
              {1, 0, 0, 0},
              {2, 1, -1, 1}},
             {"BeginNonZeroSizeMinus1Axis1",
              {3, 3, 2, 1},
              {2, 2, 1, 1},
              {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9},
              {5, 6, 8, 9},
              {1, 1, 0, 0},
              {2, -1, 1, 1}},
             {"BeginNonZeroSizeMinus1Axis2",
              {3, 2, 3, 1},
              {2, 1, 2, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 5, 5},
              {1, 0, 1, 0},
              {2, 1, -1, 1}},
             {"BeginNonZeroSizeMinus1Axis3",
              {3, 1, 2, 3},
              {2, 1, 1, 2},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 5, 5},
              {1, 0, 0, 1},
              {2, 1, 1, -1}},
             {"SliceUint8",
              {3, 2, 3, 1},
              {2, 1, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3, 5, 5, 5},
              {1, 0, 0, 0},
              {2, 1, -1, 1}},
             {"SliceInt8",
              {3, 2, 3, 1},
              {2, 1, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3, 5, 5, 5},
              {1, 0, 0, 0},
              {2, 1, -1, 1}},
             {"SliceInt16",
              {3, 2, 3, 1},
              {2, 1, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3, 5, 5, 5},
              {1, 0, 0, 0},
              {2, 1, -1, 1}},
             {"SliceInt64",
              {3, 2, 3, 1},
              {2, 1, 3, 1},
              {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6},
              {3, 3, 3, 5, 5, 5},
              {1, 0, 0, 0},
              {2, 1, -1, 1}}};

  for (auto &case_ : cases) {
    auto &name = std::get<0>(case_);
    auto &input_shape = std::get<1>(case_);
    auto &output_shape = std::get<2>(case_);
    auto &input_data = std::get<3>(case_);
    auto &output_data = std::get<4>(case_);
    auto &begin = std::get<5>(case_);
    auto &size = std::get<6>(case_);

    std::cout << name << std::endl;
    auto *param = CreateParameter(begin, size);
    TestMain({{input_shape, input_data.data(), VAR}}, {output_shape, output_data.data()}, param, false);
    param = CreateParameter(begin, size);
    TestMain({{input_shape, input_data.data(), VAR}}, {output_shape, output_data.data()}, param, true);
  }
}  // namespace mindspore

}  // namespace mindspore::lite::opencl::test
