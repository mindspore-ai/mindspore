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
#include "utils/log_adapter.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/intrp_service.h"
#include "minddata/dataset/util/task_manager.h"
#include "minddata/dataset/util/queue.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestIntrpService : public UT::Common {
 public:
    MindDataTestIntrpService() {}

    void SetUp() {}

    TaskGroup vg_;
};

TEST_F(MindDataTestIntrpService, Test1) {
  Status rc;
  Queue<int> q(3);
  q.Register(&vg_);
  vg_.CreateAsyncTask("Test1", [&]() -> Status {
    TaskManager::FindMe()->Post();
      int v;
      Status rc;
      rc = q.PopFront(&v);
      EXPECT_TRUE(rc == StatusCode::kMDInterrupted);
      return rc;
  });
  vg_.GetIntrpService()->InterruptAll();
  vg_.join_all(Task::WaitFlag::kNonBlocking);
}

TEST_F(MindDataTestIntrpService, Test2) {
  MS_LOG(INFO) << "Test Semaphore";
  Status rc;
  WaitPost wp;
  rc = wp.Register(&vg_);
  EXPECT_TRUE(rc.IsOk());
  vg_.CreateAsyncTask("Test1", [&]() -> Status {
    TaskManager::FindMe()->Post();
      Status rc = wp.Wait();
      EXPECT_TRUE(rc == StatusCode::kMDInterrupted);
      return rc;
  });
  vg_.GetIntrpService()->InterruptAll();
  vg_.join_all(Task::WaitFlag::kNonBlocking);
}