/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <vector>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/parallel/device_matrix.h"

namespace mindspore {
namespace parallel {

class TestDeviceMatrix : public UT::Common {
 public:
  TestDeviceMatrix() {}

  void SetUp() { UT::InitPythonPath(); }

  virtual void TearDown() {}
};

TEST_F(TestDeviceMatrix, Test2Dgroup_list) {
  RankList dev_list = {0, 1, 2, 3, 4, 5};
  Shape shape = {2, 3};

  DeviceMatrix arr(0, dev_list, shape);
  std::vector<RankList> group_list;
  if (arr.CreateGroupList() == Status::SUCCESS) group_list = arr.group_list();
  std::vector<RankList> group_list_expect = {{0, 3}, {0, 1, 2}};
  ASSERT_EQ(group_list, group_list_expect);
}

TEST_F(TestDeviceMatrix, Test3Dgroup_list) {
  RankList dev_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  Shape shape = {2, 2, 3};

  DeviceMatrix arr(5, dev_list, shape);
  std::vector<RankList> group_list;
  if (arr.CreateGroupList() == Status::SUCCESS) group_list = arr.group_list();
  std::vector<RankList> group_list_expect = {{5, 11}, {2, 5}, {3, 4, 5}};
  ASSERT_EQ(group_list, group_list_expect);
}

TEST_F(TestDeviceMatrix, Test4DGetAlongDim) {
  RankList dev_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  Shape shape = {2, 1, 4, 2};

  DeviceMatrix arr(5, dev_list, shape);
  std::vector<RankList> group_list;
  if (arr.CreateGroupList() == Status::SUCCESS) group_list = arr.group_list();
  std::vector<RankList> group_list_expect = {{5, 13}, {5}, {1, 3, 5, 7}, {4, 5}};
  ASSERT_EQ(group_list, group_list_expect);
}

TEST_F(TestDeviceMatrix, Test5DGetAlongDim) {
  RankList dev_list;
  for (int i = 0; i < 144; i++) dev_list.push_back(i);
  Shape shape = {3, 4, 2, 3, 2};

  DeviceMatrix arr(5, dev_list, shape);
  std::vector<RankList> group_list;
  if (arr.CreateGroupList() == Status::SUCCESS) group_list = arr.group_list();
  std::vector<RankList> group_list_expect = {{5, 53, 101}, {5, 17, 29, 41}, {5, 11}, {1, 3, 5}, {4, 5}};
  ASSERT_EQ(group_list, group_list_expect);
}

TEST_F(TestDeviceMatrix, TestCornerCaseGetAlongDim) {
  // Shape does not match the number of devices
  RankList dev_list = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  Shape shape = {2, 2, 2};

  EXPECT_THROW({ DeviceMatrix arr(3, dev_list, shape); }, std::runtime_error);
}

TEST_F(TestDeviceMatrix, TestGetDeviceByTensorMapRandomOrderSliceOne) {
  RankList dev_list = {10, 3, 2, 9, 11, 100, 1, 0};
  Shape tensor_map = {-1, 0};
  RankList rank_list;
  Shape shape = {4, 2};
  DeviceMatrix arr(0, dev_list, shape);
  arr.GetDevicesByTensorMap(tensor_map, &rank_list);
  RankList rank_list_except = {3, 9, 100, 0};
  ASSERT_EQ(rank_list, rank_list_except);
}

TEST_F(TestDeviceMatrix, TestGetDeviceByTensorMapRandomOrderSliceTwo) {
  RankList dev_list = {10, 3, 2, 9, 11, 100, 1, 0};
  Shape tensor_map = {1, 0};
  RankList rank_list;
  Shape shape = {4, 2};
  DeviceMatrix arr(0, dev_list, shape);
  arr.GetDevicesByTensorMap(tensor_map, &rank_list);
  RankList rank_list_except = {0};
  ASSERT_EQ(rank_list, rank_list_except);
}

TEST_F(TestDeviceMatrix, TestGetDeviceByTensorMapNoramalOrder2D) {
  RankList dev_list = {0, 1, 2, 3, 4, 5, 6, 7};
  Shape tensor_map = {-1, 0};
  RankList rank_list;
  Shape shape = {4, 2};
  DeviceMatrix arr(6, dev_list, shape);
  arr.GetDevicesByTensorMap(tensor_map, &rank_list);
  RankList rank_list_except = {0, 2, 4, 6};
  ASSERT_EQ(rank_list, rank_list_except);
}

TEST_F(TestDeviceMatrix, TestCornerCase2GetAlongDim) {
  // Rank is out of range
  RankList dev_list = {0, 1, 2, 3, 4, 5, 6, 7};
  Shape shape = {2, 2, 2};

  EXPECT_THROW({ DeviceMatrix arr(8, dev_list, shape); }, std::runtime_error);
}

}  // namespace parallel
}  // namespace mindspore
