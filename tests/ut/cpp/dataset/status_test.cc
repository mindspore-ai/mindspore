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
#include "minddata/dataset/util/status.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestStatus : public UT::Common {
 public:
    MindDataTestStatus() {}
};

// This function returns Status
Status f1() {
  Status rc(StatusCode::kMDUnexpectedError, "Testing macro");
  RETURN_IF_NOT_OK(rc);
  // We shouldn't get here
  return Status::OK();
}

Status f3() {
  RETURN_STATUS_UNEXPECTED("Testing macro3");
}

TEST_F(MindDataTestStatus, Test1) {
  // Test default constructor which should be OK
  Status rc;
  ASSERT_TRUE(rc.IsOk());
  Status err1(StatusCode::kMDOutOfMemory, __LINE__, __FILE__);
  MS_LOG(DEBUG) << err1;
  ASSERT_TRUE(err1 == StatusCode::kMDOutOfMemory);
  ASSERT_TRUE(err1.IsError());
  Status err2(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "Oops");
  MS_LOG(DEBUG) << err2;
}

TEST_F(MindDataTestStatus, Test2) {
  Status rc = f1();
  MS_LOG(DEBUG) << rc;
}

TEST_F(MindDataTestStatus, Test3) {
  Status rc = f3();
  MS_LOG(DEBUG) << rc;
}
