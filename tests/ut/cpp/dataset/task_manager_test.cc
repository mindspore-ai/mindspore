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
#include "dataset/util/task_manager.h"

using namespace mindspore::dataset;
using namespace std::placeholders;

class MindDataTestTaskManager : public UT::Common {
 public:
    MindDataTestTaskManager() {}

    void SetUp() { Services::CreateInstance();
    }
};

std::atomic<int> v(0);

Status f(TaskGroup &vg){
  for (int i = 0; i < 1; i++) {
    RETURN_IF_NOT_OK(vg.CreateAsyncTask("Infinity", [&]() -> Status {
        TaskManager::FindMe()->Post();
        int a = v.fetch_add(1);
        MS_LOG(DEBUG) << a << std::endl;
        return f(vg);
    }));
  }
  return Status::OK();
}

TEST_F(MindDataTestTaskManager, Test1) {
  // Clear the rc of the master thread if any
  (void) TaskManager::GetMasterThreadRc();
  TaskGroup vg;
  Status vg_rc = vg.CreateAsyncTask("Test error", [this]() -> Status {
    TaskManager::FindMe()->Post();
    throw std::bad_alloc();
  });
  ASSERT_TRUE(vg_rc.IsOk() || vg_rc.IsOutofMemory());
  ASSERT_TRUE(vg.join_all().IsOk());
  ASSERT_TRUE(vg.GetTaskErrorIfAny().IsOutofMemory());
  // Test the error is passed back to the master thread.
  Status rc = TaskManager::GetMasterThreadRc();
  ASSERT_TRUE(rc.IsOutofMemory());
}
