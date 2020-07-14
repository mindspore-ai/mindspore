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
#include "frontend/parallel/device_manager.h"
#include "common/common_test.h"
#include "frontend/parallel/device.h"
#include "frontend/parallel/group_manager.h"

namespace mindspore {
namespace parallel {

extern DeviceManagerPtr g_device_manager;

class TestGroup : public UT::Common {
 public:
  TestGroup() {}
  void SetUp();
  void TearDown();
  Status Init();

  Group gp;
};

void TestGroup::SetUp() { gp = Group(); }

void TestGroup::TearDown() {
  // destroy resources
}

Status TestGroup::Init() {
  std::string gname = "1-2";
  std::vector<Device> dev_list;
  Device one = Device(int32_t(1));
  dev_list.push_back(one);
  Device two = Device(int32_t(2));
  dev_list.push_back(two);

  return gp.Init(gname, dev_list);
}

TEST_F(TestGroup, test_Init) { ASSERT_EQ(Init(), Status::SUCCESS); }

TEST_F(TestGroup, test_GetDevicesList) {
  Init();
  std::vector<Device> res_dev_list = gp.GetDevicesList();
  std::vector<Device>::iterator it = res_dev_list.begin();
  ASSERT_EQ(it->rank(), int32_t(1));
  it++;
  ASSERT_EQ(it->rank(), int32_t(2));
}

TEST_F(TestGroup, test_IsInThisGroup) {
  Init();
  ASSERT_TRUE(gp.IsInThisGroup(int32_t(1)));
  ASSERT_TRUE(gp.IsInThisGroup(int32_t(2)));

  ASSERT_FALSE(gp.IsInThisGroup(int32_t(3)));
}

class TestGroupManager : public UT::Common {
 public:
  TestGroupManager() {}
  void SetUp();
  void TearDown();
  Status Init(Group** gp_ptr);

  GroupManager gm;
};

void TestGroupManager::SetUp() { gm = GroupManager(); }

void TestGroupManager::TearDown() {
  // destroy resources
}

Status TestGroupManager::Init(Group** gp_ptr) {
  std::string gname = "1-2";
  std::vector<Device> dev_list;
  Device one = Device(int32_t(1));
  dev_list.push_back(one);
  Device two = Device(int32_t(2));
  dev_list.push_back(two);

  return gm.CreateGroup(gname, dev_list, *gp_ptr);
}

TEST_F(TestGroupManager, test_CreateGroup) {
  // testing for creating a group
  Group* gp_ptr = new Group();
  ASSERT_EQ(Init(&gp_ptr), Status::SUCCESS);

  std::vector<Device> res_dev_list = gp_ptr->GetDevicesList();
  std::vector<Device>::iterator it = res_dev_list.begin();
  ASSERT_EQ(it->rank(), int32_t(1));
  it++;
  ASSERT_EQ(it->rank(), int32_t(2));
  delete gp_ptr;

  // testing for creating a group with an existing group name
  std::vector<Device> dev_list2;
  Device three = Device(int32_t(3));
  dev_list2.push_back(three);
  Device four = Device(int32_t(4));
  dev_list2.push_back(four);
  gp_ptr = new Group();
  ASSERT_EQ(gm.CreateGroup("1-2", dev_list2, gp_ptr), Status::SUCCESS);

  ASSERT_STREQ(gp_ptr->name().data(), "1-2");
  std::vector<Device> res_dev_list2 = gp_ptr->GetDevicesList();
  std::vector<Device>::iterator it2 = res_dev_list2.begin();
  ASSERT_EQ(it2->rank(), int32_t(1));
  it2++;
  ASSERT_EQ(it2->rank(), int32_t(2));
  delete gp_ptr;
  gp_ptr = nullptr;
}

TEST_F(TestGroupManager, test_FindGroup) {
  std::string gname = "1-2";
  Group* gp_ptr = new Group();
  Group* gp_ptr2 = new Group();
  ASSERT_EQ(Init(&gp_ptr), Status::SUCCESS);

  ASSERT_EQ(gm.FindGroup(gname, &gp_ptr2), Status::SUCCESS);

  std::vector<Device> res_dev_list = gp_ptr2->GetDevicesList();
  std::vector<Device>::iterator it = res_dev_list.begin();
  ASSERT_EQ(it->rank(), int32_t(1));
  it++;
  ASSERT_EQ(it->rank(), int32_t(2));
  delete gp_ptr;
  gp_ptr = nullptr;

  std::string gname2 = "3-4";
  gp_ptr2 = new Group();
  ASSERT_EQ(gm.FindGroup(gname2, &gp_ptr2), Status::FAILED);
  delete gp_ptr2;
  gp_ptr2 = nullptr;
}

}  // namespace parallel
}  // namespace mindspore
