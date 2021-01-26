/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <memory>
#include "common/common_test.h"
#define private public
#include "include/api/status.h"
#undef private

namespace mindspore {
class TestCxxApiStatus : public UT::Common {
 public:
  TestCxxApiStatus() = default;
};

TEST_F(TestCxxApiStatus, test_status_base_SUCCESS) {
  Status status_1;
  ASSERT_TRUE(status_1 == kSuccess);
  ASSERT_TRUE(status_1 == Status(kSuccess));
  ASSERT_EQ(status_1.operator bool(), true);
  ASSERT_EQ(status_1.operator int(), kSuccess);
  ASSERT_EQ(status_1.StatusCode(), kSuccess);
  ASSERT_EQ(status_1.IsOk(), true);
  ASSERT_EQ(status_1.IsError(), false);
}

TEST_F(TestCxxApiStatus, test_status_msg_SUCCESS) {
  std::string message = "2333";
  Status status_1(kMDSyntaxError, message);
  ASSERT_EQ(status_1.IsError(), true);
  ASSERT_EQ(status_1.ToString(), message);
}

TEST_F(TestCxxApiStatus, test_status_ctor_SUCCESS) {
  Status status_1;
  Status status_2(kSuccess);
  Status status_3(kSuccess, "2333");
  Status status_4(kSuccess, 1, "file", "2333");
  Status status_5 = Status::OK();
  ASSERT_TRUE(status_1 == status_2);
  ASSERT_TRUE(status_1 == status_3);
  ASSERT_TRUE(status_1 == status_4);
  ASSERT_TRUE(status_1 == status_5);
}

TEST_F(TestCxxApiStatus, test_status_string_SUCCESS) {
  Status status_1(kMDSyntaxError);
  ASSERT_EQ(Status::CodeAsString(status_1.StatusCode()), "Syntax error");
}
}  // namespace mindspore
