/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "minddata/dataset/engine/perf/cyclic_array.h"
#include "minddata/dataset/engine/perf/perf_data.h"

using namespace mindspore::dataset;

class MindDataTestPerfData : public UT::Common {
 public:
  MindDataTestPerfData() {}
};

/// Feature: PerfData
/// Description: Test PerfData int CyclicArray and std::vector<int64_t> with AddSample by comparing them together
/// Expectation: Both CyclicArray and vector should be equal
TEST_F(MindDataTestPerfData, Test1) {
  PerfData<std::vector<int64_t>> p1(2, 3);
  PerfData<CyclicArray<int>> p2(2, 3);
  EXPECT_EQ(p1.capacity(), p2.capacity());
  std::vector<int64_t> row = {1, 2, 3};
  p1.AddSample(row);
  p2.AddSample(row);
  EXPECT_EQ(p1.size(), 1);
  EXPECT_EQ(p1.size(), p2.size());
  p1.AddSample(row);
  p2.AddSample(row);
  EXPECT_EQ(p1.size(), 2);
  EXPECT_EQ(p1.size(), p2.size());
  std::vector<int64_t> row1 = {4, 5, 6};
  p2.AddSample(row1);
  EXPECT_EQ(p2.size(), 2);
  auto r1 = p2.Row<int>(static_cast<int64_t>(0));
  for (auto i = 0; i < 3; i++) {
    EXPECT_EQ(r1[i], i + 1);
  }

  auto r2 = p2.Row<int>(1);
  for (auto i = 0; i < 3; i++) {
    EXPECT_EQ(r2[i], i + 4);
  }

  EXPECT_EQ(p2[0][1], 4);
  EXPECT_EQ(p2[1][1], 5);
  EXPECT_EQ(p2[2][1], 6);
}

/// Feature: PerfData
/// Description: Test PerfData int CyclicArray by using AddSample to add rows to the CyclicArray
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPerfData, Test2) {
  auto pd = PerfData<CyclicArray<int>>(1000000, 3);
  auto row = {1, 2, 3};
  pd.AddSample(row);
  EXPECT_EQ(pd[0][0], 1);
  EXPECT_EQ(pd[1][0], 2);
  EXPECT_EQ(pd[2][0], 3);
  auto row1 = {4, 5, 6};
  pd.AddSample(row1);
  EXPECT_EQ(pd[0][0], 1);
  EXPECT_EQ(pd[1][0], 2);
  EXPECT_EQ(pd[2][0], 3);
  EXPECT_EQ(pd[0][1], 4);
  EXPECT_EQ(pd[1][1], 5);
  EXPECT_EQ(pd[2][1], 6);
}