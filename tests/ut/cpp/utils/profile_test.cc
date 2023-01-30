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
#include <iostream>
#include <string>
#include "common/common_test.h"

#ifndef ENABLE_PROFILE
#define ENABLE_PROFILE
#endif

#include "utils/profile.h"

namespace mindspore {
class TestProfile : public UT::Common {
 public:
  TestProfile() {}
  virtual ~TestProfile() {}

  virtual void TearDown() {}
};

static void test_lap(Profile *prof) {
  int nums[] = {30, 20, 70};
  int cnt = 0;
  for (auto elem : nums) {
    ProfileExecute(prof->Lap(cnt), [elem]() -> void { usleep(elem); });
    cnt += 1;
  }
}

TEST_F(TestProfile, Test01) {
  int step_cnt = 0;
  Profile prof;
  Profile *ptr_prof = &prof;
  DumpTime::GetInstance().Record("Test01", GetTime(), true);
  ProfileExecute(ptr_prof, [&ptr_prof, &step_cnt]() -> void {
    ProfileExecute(ptr_prof->Step("Step01"), [&step_cnt]() -> void {
      usleep(20);
      step_cnt += 1;
    });
    ProfileExecute(ptr_prof->Step("Step02"), [&ptr_prof, &step_cnt]() -> void {
      usleep(10);
      test_lap(ptr_prof);
      step_cnt += 1;
    });
  });
  DumpTime::GetInstance().Record("Test01", GetTime(), false);

  prof.Print();

  EXPECT_EQ(step_cnt, 2);
}

TEST_F(TestProfile, Test02) {
  std::map<std::string, TimeStat> stat;
  double t1 = GetTime();
  usleep(20);  // Step01.stage1
  double t2 = GetTime();
  usleep(30);  // Step01.stage2
  double t3 = GetTime();
  usleep(10);  // Step02.stage1
  double t4 = GetTime();
  usleep(10);  // Step02.stage2
  double t5 = GetTime();
  usleep(10);  // Step02.stage3
  double t6 = GetTime();

  MsProfile::StatTime("Step01.stage1", t2 - t1);
  MsProfile::StatTime("Step01.stage2", t3 - t2);
  MsProfile::StatTime("Step02.stage1", t4 - t3);
  MsProfile::StatTime("Step02.stage2", t5 - t4);
  MsProfile::StatTime("Step02.stage3", t6 - t5);

  MsProfile::Print();
  MsProfile::Reset();
  EXPECT_GT(t6 - t1, 0);
}

}  // namespace mindspore
