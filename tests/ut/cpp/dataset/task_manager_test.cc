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

#include "common/common.h"
#include "gtest/gtest.h"
#include "minddata/dataset/util/task_manager.h"

using namespace mindspore::dataset;

class MindDataTestTaskManager : public UT::Common {
 public:
  MindDataTestTaskManager() {}

  void SetUp() { Services::CreateInstance(); }
};

TEST_F(MindDataTestTaskManager, Test1) {
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  TaskGroup vg;
  Status vg_rc = vg.CreateAsyncTask("Test error", []() -> Status {
    TaskManager::FindMe()->Post();
    throw std::bad_alloc();
  });
  ASSERT_TRUE(vg_rc.IsOk() || vg_rc == StatusCode::kMDOutOfMemory);
  ASSERT_TRUE(vg.join_all().IsOk());
  ASSERT_TRUE(vg.GetTaskErrorIfAny() == StatusCode::kMDOutOfMemory);
  // Test the error is passed back to the master thread if vg_rc above is OK.
  // If vg_rc is kOutOfMemory, the group error is already passed back.
  // Some compiler may choose to run the next line in parallel with the above 3 lines
  // and this will cause some mismatch once a while.
  // To block this racing condition, we need to create a dependency that the next line
  // depends on previous lines.
  if (vg.GetTaskErrorIfAny().IsError() && vg_rc.IsOk()) {
    Status rc = TaskManager::GetMasterThreadRc();
    ASSERT_TRUE(rc == StatusCode::kMDOutOfMemory);
  }
}

TEST_F(MindDataTestTaskManager, Test2) {
  // This testcase will spawn about 100 threads and block on a conditional variable.
  // The master thread will try to interrupt them almost at the same time. This can
  // cause a racing condition that some threads may miss the interrupt and blocked.
  // The new logic of Task::Join() will do a time-out join and wake up all those
  // threads that miss the interrupt.
  // Clear the rc of the master thread if any
  (void)TaskManager::GetMasterThreadRc();
  TaskGroup vg;
  CondVar cv;
  std::mutex mux;
  Status rc;
  rc = cv.Register(vg.GetIntrpService());
  EXPECT_TRUE(rc.IsOk());
  auto block_forever = [&cv, &mux]() -> Status {
    std::unique_lock<std::mutex> lck(mux);
    TaskManager::FindMe()->Post();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    RETURN_IF_NOT_OK(cv.Wait(&lck, []() -> bool { return false; }));
    return Status::OK();
  };
  auto f = [&vg, &block_forever]() -> Status {
    for (auto i = 0; i < 100; ++i) {
      RETURN_IF_NOT_OK(vg.CreateAsyncTask("Spawn block threads", block_forever));
    }
    return Status::OK();
  };
  rc = f();
  vg.interrupt_all();
  EXPECT_TRUE(rc.IsOk());
  // Now we test the async Join
  ASSERT_TRUE(vg.join_all(Task::WaitFlag::kNonBlocking).IsOk());
}
