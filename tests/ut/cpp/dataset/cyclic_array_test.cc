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
#include <iterator>
#include <algorithm>
#include "common/common.h"
#include "common/cvop_common.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "minddata/dataset/engine/perf/cyclic_array.h"
#include <chrono>

using namespace mindspore::dataset;

class MindDataTestCyclicArray : public UT::Common {
 public:
  MindDataTestCyclicArray() {}
};

/// Feature: CyclicArray
/// Description: Test CyclicArray attributes and basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestCyclicArray, Test1) {
  CyclicArray<int> arr(5);
  EXPECT_EQ(5, arr.capacity());
  EXPECT_EQ(0, arr.size());
  arr.push_back(0);
  EXPECT_EQ(5, arr.capacity());
  EXPECT_EQ(1, arr.size());
  EXPECT_EQ(arr[0], 0);
  arr.push_back(1);
  EXPECT_EQ(arr[1], 1);
  for (auto i = 2; i < 5; i++) {
    arr.push_back(i);
  }
  EXPECT_EQ(arr.capacity(), arr.size());
  EXPECT_EQ(1, arr[1]);
  EXPECT_EQ(4, arr[4]);
  arr[4] = 42;
  EXPECT_EQ(arr[4], 42);
  auto a = arr[4];
  EXPECT_EQ(a, 42);
  arr.push_back(5);
  EXPECT_EQ(arr[0], 1);
  EXPECT_EQ(arr[4], 5);

  CyclicArray<int> arr2 = arr;
  EXPECT_EQ(arr2.capacity(), arr.capacity());
  EXPECT_EQ(arr2.size(), arr.size());
  auto last = arr2.end();
  auto first = arr2.begin();
  for (auto i = 0; i < arr.size(); i++) {
    EXPECT_EQ(arr2[i], arr[i]);
  }

  arr.clear();
  EXPECT_EQ(arr.size(), 0);
  arr.push_back(42);
  arr.push_back(43);
  EXPECT_EQ(arr.size(), 2);
  EXPECT_EQ(arr.capacity(), 5);
  EXPECT_EQ(arr[0], 42);
  EXPECT_EQ(arr[1], 43);
  auto arr3 = arr;
  EXPECT_EQ(arr3.size(), 2);
  EXPECT_EQ(arr3.capacity(), 5);
  EXPECT_EQ(arr.size(), 2);
  EXPECT_EQ(arr.capacity(), 5);
  EXPECT_EQ(arr[0], arr3[0]);
  EXPECT_EQ(arr[1], arr3[1]);

  arr.clear();
  arr.push_back(21);
  arr.push_back(22);
  EXPECT_EQ(arr[arr.size() - 1], 22);
  for (auto i = 23; i < 27; i++) {
    arr.push_back(i);
  }
  EXPECT_EQ(arr[0], 22);
  EXPECT_EQ(arr[arr.size() - 1], 26);
}

/// Feature: CyclicArray
/// Description: Test iterating over CyclicArray
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestCyclicArray, TestIterator) {
  CyclicArray<int> arr(5);
  for (auto i = 0; i < arr.capacity(); i++) {
    arr.push_back(i);
  }
  arr.push_back(6);
  arr.push_back(7);
  auto i = 0;
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    EXPECT_EQ(*it, arr[i++]);
  }

  std::iota(arr.begin(), arr.end(), -4);
  EXPECT_EQ(arr[0], -4);
  EXPECT_EQ(arr[4], 0);
  const auto sz = 1000000;
  CyclicArray<int> arr2(sz);
  for (auto i = 0; i < sz - 1; i++) {
    arr.push_back(0);
  }
  const auto val = -500000;
  std::iota(arr2.begin(), arr2.end() + sz, val);
  EXPECT_EQ(*arr2.begin(), val);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(arr2.begin(), arr2.end(), g);
  std::sort(arr2.begin(), arr2.end(), [](const auto a, const auto b) { return a > b; });
  EXPECT_EQ(*arr2.begin(), val);
  const auto new_val = -600000;
  for (auto i = 0; i < 100; i++) {
    arr2.push_back(new_val);
  }
  EXPECT_EQ(*(--arr2.end()), new_val);
  std::sort(arr2.begin(), arr2.end(), [](const auto a, const auto b) { return a > b; });
  EXPECT_EQ(*arr2.begin(), new_val);
}
