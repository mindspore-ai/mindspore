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
#include "nnacl/strided_slice_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_StridedSlice : public CommonTest {};

namespace {
// PrimitiveType_StridedSlice: src/ops/populate/strided_slice_populate.cc
OpParameter *CreateParameter(const std::vector<int> &begins, const std::vector<int> &ends,
                             const std::vector<int> &strides) {
  auto *param = test::CreateParameter<StridedSliceParameter>(schema::PrimitiveType_StridedSlice);
  param->num_axes_ = begins.size();
  for (int i = 0; i < begins.size(); ++i) {
    param->begins_[i] = begins[i];
    param->ends_[i] = ends[i];
    param->strides_[i] = strides[i];
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_StridedSlice, 1D) {
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  float output_data[] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({3}, {36}, {3});
    TestMain({{{36}, input_data, VAR}}, {{11}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_StridedSlice, 2D) {
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  float output_data[] = {11, 14};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({1, 2}, {3, 8}, {2, 3});
    TestMain({{{4, 9}, input_data, VAR}}, {{1, 2}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_StridedSlice, 3D) {
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  float output_data[] = {11, 14};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({0, 1, 2}, {1, 3, 8}, {1, 2, 3});
    TestMain({{{1, 4, 9}, input_data, VAR}}, {{1, 1, 2}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_StridedSlice, 4D) {
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};

  float output_data0[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({0, 0, 0, 0}, {2, 2, 3, 3}, {1, 1, 1, 1});
    TestMain({{{2, 2, 3, 3}, input_data, VAR}}, {{2, 2, 3, 3}, output_data0}, param, fp16_enable);
  }

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({0, 0, 0, 0}, {2, 2, 3, 3}, {1, 1, 1, 1});
    TestMain({{{2, 2, 3, 3}, input_data, VAR}}, {{2, 2, 3, 3}, output_data0}, param, fp16_enable);
  }

  float output_data1[] = {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({1, 0, 0, 0}, {2, 2, 3, 3}, {1, 1, 1, 1});
    TestMain({{{2, 2, 3, 3}, input_data, VAR}}, {{1, 2, 3, 3}, output_data1}, param, fp16_enable);
  }

  float output_data2[] = {27, 28, 29, 30, 31, 32, 33, 34, 35};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({1, 1, 0, 0}, {2, 2, 3, 3}, {1, 1, 1, 1});
    TestMain({{{2, 2, 3, 3}, input_data, VAR}}, {{1, 1, 3, 3}, output_data2}, param, fp16_enable);
  }

  float output_data3[] = {33, 34, 35};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({1, 1, 2, 0}, {2, 2, 3, 3}, {1, 1, 1, 1});
    TestMain({{{2, 2, 3, 3}, input_data, VAR}}, {{1, 1, 1, 3}, output_data3}, param, fp16_enable);
  }

  float output_data4[] = {34};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({1, 1, 2, 1}, {2, 2, 3, 2}, {1, 1, 1, 1});
    TestMain({{{2, 2, 3, 3}, input_data, VAR}}, {{1, 1, 1, 1}, output_data4}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_StridedSlice, 4D_stride2) {
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  float output_data[] = {13, 14, 31, 32};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({0, 1, 1, 1}, {1, 4, 3, 3}, {2, 2, 2, 1});
    TestMain({{{1, 4, 3, 3}, input_data, VAR}}, {{1, 2, 1, 2}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_StridedSlice, 4D_to_3D) {
  float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
  float output_data[] = {18, 20, 21, 23, 27, 29, 30, 32};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({1, 0, 0, 0}, {2, 2, 2, 3}, {1, 1, 1, 2});
    TestMain({{{2, 2, 3, 3}, input_data, VAR}}, {{2, 2, 2}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_StridedSlice, In1D_OutOfRangeBeginNegativeStride) {
  float input_data[] = {1, 2, 3, 4};
  float output_data[] = {4, 3, 2};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({5}, {0}, {-1});
    TestMain({{{4}, input_data, VAR}}, {{3}, output_data}, param, fp16_enable);
  }
}

TEST_F(TestOpenCL_StridedSlice, test0) {
  std::vector<float> values(32768);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = i % 1000;
  }
  std::vector<std::tuple<std::string, std::vector<int>, std::vector<int>, std::vector<float>, std::vector<float>,
                         std::vector<int>, std::vector<int>, std::vector<int>>>
    cases = {{"In1D", {4}, {2}, {1, 2, 3, 4}, {2, 3}, {1}, {3}, {1}},
             {"In1D_Int32End", {32768}, {32768}, values, values, {0}, {32768}, {1}},
             {"In1D_NegativeBegin", {4}, {2}, {1, 2, 3, 4}, {2, 3}, {-3}, {3}, {1}},
             {"In1D_OutOfRangeBegin", {4}, {3}, {1, 2, 3, 4}, {1, 2, 3}, {-5}, {3}, {1}},
             {"In1D_NegativeEnd", {4}, {1}, {1, 2, 3, 4}, {2}, {1}, {-2}, {1}},
             {"In1D_OutOfRangeEnd", {4}, {3}, {1, 2, 3, 4}, {2, 3, 4}, {-3}, {5}, {1}},
             {"In1D_NegativeBeginNegativeStride", {4}, {1}, {1, 2, 3, 4}, {3}, {-2}, {-3}, {-1}},
             {"In1D_OutOfRangeBeginNegativeStride", {4}, {1}, {1, 2, 3, 4}, {4}, {5}, {2}, {-1}},
             {"In1D_NegativeEndNegativeStride", {4}, {2}, {1, 2, 3, 4}, {3, 2}, {2}, {-4}, {-1}},
             {"In1D_OutOfRangeEndNegativeStride", {4}, {2}, {1, 2, 3, 4}, {2, 1}, {-3}, {-5}, {-1}},
             {"In1D_NegStride", {3}, {3}, {1, 2, 3}, {3, 2, 1}, {-1}, {-4}, {-1}},
             {"In1D_EvenLenStride2", {2}, {1}, {1, 2}, {1}, {0}, {2}, {2}},
             {"In1D_OddLenStride2", {3}, {2}, {1, 2, 3}, {1, 3}, {0}, {3}, {2}},
             {"In2D_Identity", {2, 3}, {2, 3}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, {0, 0}, {2, 3}, {1, 1}},
             {"In2D", {2, 3}, {1, 2}, {1, 2, 3, 4, 5, 6}, {4, 5}, {1, 0}, {2, 2}, {1, 1}},
             {"In2D_Stride2", {2, 3}, {1, 2}, {1, 2, 3, 4, 5, 6}, {1, 3}, {0, 0}, {2, 3}, {2, 2}},
             {"In2D_NegStride", {2, 3}, {1, 3}, {1, 2, 3, 4, 5, 6}, {6, 5, 4}, {1, -1}, {2, -4}, {2, -1}},
             {"In2D_BeginMask", {2, 3}, {2, 2}, {1, 2, 3, 4, 5, 6}, {1, 2, 4, 5}, {0, 0}, {2, 2}, {1, 1}},
             {"In2D_EndMask", {2, 3}, {1, 3}, {1, 2, 3, 4, 5, 6}, {4, 5, 6}, {1, 0}, {2, 3}, {1, 1}},
             {"In2D_NegStrideBeginMask", {2, 3}, {1, 3}, {1, 2, 3, 4, 5, 6}, {6, 5, 4}, {1, -1}, {2, -4}, {1, -1}},
             {"In2D_NegStrideEndMask", {2, 3}, {1, 2}, {1, 2, 3, 4, 5, 6}, {6, 5}, {1, -1}, {2, 0}, {1, -1}},
             {"In3D_Identity",
              {2, 3, 2},
              {2, 3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {0, 0, 0},
              {2, 3, 2},
              {1, 1, 1}},
             {"In3D_NegStride",
              {2, 3, 2},
              {2, 3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
              {-1, -1, -1},
              {-3, -4, -3},
              {-1, -1, -1}},
             {"In3D_Strided2",
              {2, 3, 2},
              {1, 2, 1},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 5},
              {0, 0, 0},
              {2, 3, 2},
              {2, 2, 2}},
             {"In1D_ShrinkAxisMask1", {4}, {1}, {1, 2, 3, 4}, {2}, {1}, {2}, {1}},
             {"In1D_ShrinkAxisMask1_NegativeSlice", {4}, {1}, {0, 1, 2, 3}, {3}, {-1}, {4}, {1}},
             {"In2D_ShrinkAxis3_NegativeSlice", {4, 1}, {1}, {0, 1, 2, 3}, {2}, {-2, -1}, {3, 1}, {1, 1}},
             {"In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice",
              {4, 1},
              {4},
              {0, 1, 2, 3},
              {0, 1, 2, 3},
              {0, -1},
              {4, 1},
              {1, 1}},
             {"In1D_BeginMaskShrinkAxisMask1", {4}, {1}, {1, 2, 3, 4}, {1}, {0}, {1}, {1}},
             {"In2D_ShrinkAxisMask1", {2, 3}, {3}, {1, 2, 3, 4, 5, 6}, {1, 2, 3}, {0, 0}, {1, 3}, {1, 1}},
             {"In2D_ShrinkAxisMask2", {2, 3}, {2}, {1, 2, 3, 4, 5, 6}, {1, 4}, {0, 0}, {2, 1}, {1, 1}},
             {"In2D_ShrinkAxisMask3", {2, 3}, {1}, {1, 2, 3, 4, 5, 6}, {1}, {0, 0}, {1, 1}, {1, 1}},
             {"In3D_IdentityShrinkAxis1",
              {2, 3, 2},
              {3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 3, 4, 5, 6},
              {0, 0, 0},
              {1, 3, 2},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis2",
              {2, 3, 2},
              {2, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 7, 8},
              {0, 0, 0},
              {2, 1, 2},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis3",
              {2, 3, 2},
              {2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2},
              {0, 0, 0},
              {1, 1, 2},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis4",
              {2, 3, 2},
              {2, 3},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 3, 5, 7, 9, 11},
              {0, 0, 0},
              {2, 3, 1},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis5",
              {2, 3, 2},
              {3},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 3, 5},
              {0, 0, 0},
              {1, 3, 1},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis6",
              {2, 3, 2},
              {2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 7},
              {0, 0, 0},
              {2, 1, 1},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis7",
              {2, 3, 2},
              {1},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1},
              {0, 0, 0},
              {1, 1, 1},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis1Uint8",
              {2, 3, 2},
              {3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 3, 4, 5, 6},
              {0, 0, 0},
              {1, 3, 2},
              {1, 1, 1}},
             {"In3D_IdentityShrinkAxis1int8",
              {2, 3, 2},
              {3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 3, 4, 5, 6},
              {0, 0, 0},
              {1, 3, 2},
              {1, 1, 1}},
             {"In5D_Identity",
              {2, 2, 2, 2},
              {2, 1, 2, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
              {1, 2, 3, 4, 9, 10, 11, 12},
              {0, 0, 0, 0},
              {2, 1, 2, 2},
              {1, 1, 1, 1}},
             {"In5D_IdentityShrinkAxis1",
              {2, 2, 2, 2},
              {1, 2, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
              {1, 2, 3, 4},
              {0, 0, 0, 0},
              {1, 1, 2, 2},
              {1, 1, 1, 1}},
             {"In3D_SmallBegin",
              {2, 3, 2},
              {1, 3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 3, 4, 5, 6},
              {0},
              {1},
              {1}},
             {"In3D_SmallBeginWithhrinkAxis1",
              {2, 3, 2},
              {3, 2},
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              {1, 2, 3, 4, 5, 6},
              {0},
              {1},
              {1}}};

  for (auto &case_ : cases) {
    auto &name = std::get<0>(case_);
    auto &input_shape = std::get<1>(case_);
    auto &output_shape = std::get<2>(case_);
    auto input_data = std::get<3>(case_).data();
    auto output_data = std::get<4>(case_).data();
    auto &begin = std::get<5>(case_);
    auto &end = std::get<6>(case_);
    auto &stride = std::get<7>(case_);
    std::cout << name << std::endl;

    for (auto fp16_enable : {false, true}) {
      auto *param = CreateParameter(begin, end, stride);
      TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
    }
  }
}

TEST_F(TestOpenCL_StridedSlice, test1) {
  float input_data[] = {0.1,  0.2,  0.3,  0.4,  1.1,  1.2,  1.3,  1.4,  10.1, 10.2, 10.3, 10.4,
                        11.1, 11.2, 11.3, 11.4, 20.1, 20.2, 20.3, 20.4, 21.1, 21.2, 21.3, 21.4};
  float output_data[] = {10.2, 10.4, 20.2, 20.4};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter({0, 1, 0, 1}, {1, 3, 2, 4}, {1, 1, 2, 2});
    TestMain({{{1, 3, 2, 4}, input_data, VAR}}, {{1, 2, 1, 2}, output_data}, param, fp16_enable,
             fp16_enable ? 1e-2 : 1e-9);
  }
}

}  // namespace mindspore::lite::opencl::test
