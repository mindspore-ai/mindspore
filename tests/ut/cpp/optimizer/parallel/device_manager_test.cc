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
#include <list>
#include "common/common_test.h"
#include "optimizer/parallel/device.h"
#include "optimizer/parallel/device_manager.h"
#include "optimizer/parallel/group_manager.h"

namespace mindspore {
namespace parallel {

class TestDevice : public UT::Common {
 public:
  TestDevice() {}
  void SetUp();
  void TearDown();
  Device dev_1;
  Device dev_2;
};

void TestDevice::SetUp() {
  std::string name = "#1";
  dev_1 = Device(name, std::int32_t(1));
  dev_2 = Device(std::int32_t(2));
}

void TestDevice::TearDown() {
  // destroy resources
}

TEST_F(TestDevice, test_device) {
  std::string name = "#1";
  int32_t dev1_rank = 1;
  int32_t dev2_rank = 2;

  ASSERT_STREQ(dev_1.name().data(), name.data());
  ASSERT_EQ(dev_1.rank(), dev1_rank);
  ASSERT_EQ(dev_2.rank(), dev2_rank);
}

// need to complete
class TestStage : public UT::Common {};

class TestDeviceManager : public UT::Common {
 public:
  TestDeviceManager() {}
  void SetUp();
  void TearDown();
  DeviceManager dm_;
};

void TestDeviceManager::SetUp() { dm_ = DeviceManager::GetInstance(); }

void TestDeviceManager::TearDown() {
  // destroy resources
}

TEST_F(TestDeviceManager, test_dm_init_AND_get_device_list) {
  std::list<int32_t> dev_list;
  std::list<int32_t> stage_map;
  int32_t local_dev = 0;

  dev_list.push_back(5);
  dev_list.push_back(3);
  dev_list.push_back(1);
  dev_list.push_back(0);

  stage_map.push_back(2);
  stage_map.push_back(2);
  ASSERT_EQ(dm_.Init(dev_list, local_dev, stage_map, "hccl"), Status::SUCCESS);

  ASSERT_EQ(dm_.DeviceNum(), 4);
  ASSERT_EQ(dm_.GetStageNum(), (int32_t)(2));

  std::list<int32_t> dev_list_0 = dm_.GetDeviceListByStageId(0);
  std::list<int32_t> dev_list_1 = dm_.GetDeviceListByStageId(1);
  ASSERT_EQ(dev_list_0.size(), 2);
  ASSERT_EQ(dev_list_1.size(), 2);

  std::list<int32_t>::iterator it = dev_list_0.begin();
  ASSERT_EQ((*it), int32_t(5));
  it++;
  ASSERT_EQ((*it), int32_t(3));
  it = dev_list_1.begin();
  ASSERT_EQ((*it), int32_t(1));
  it++;
  ASSERT_EQ((*it), int32_t(0));

  std::shared_ptr<Stage> stage_0 = dm_.GetStageById(0);
  ASSERT_EQ(stage_0->GetDevicesNum(), size_t(2));
  std::shared_ptr<Stage> stage_1 = dm_.GetStageById(1);
  ASSERT_EQ(stage_1->GetDevicesNum(), size_t(2));
}

TEST_F(TestDeviceManager, test_CreateNewDeviceByRank) {
  Device one = dm_.CreateNewDeviceByRank(int32_t(3));
  ASSERT_EQ(one.rank(), int32_t(3));
}

TEST_F(TestDeviceManager, test_CreateDeviceListByRankList) {
  std::list<Device> dev_list;
  std::list<int32_t> rlist;
  rlist.push_back(int32_t(2));
  rlist.push_back(int32_t(1));
  dev_list = dm_.CreateDeviceListByRankList(rlist);

  std::list<Device>::iterator it = dev_list.begin();
  ASSERT_EQ(it->rank(), int32_t(2));
  it++;
  ASSERT_EQ(it->rank(), int32_t(1));
}

}  // namespace parallel
}  // namespace mindspore
