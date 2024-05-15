/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "common/common_test.h"
#include "utils/log_adapter.h"

namespace {
constexpr char kMsExceptionDisplayLevel[] = "MS_EXCEPTION_DISPLAY_LEVEL";
}  // namespace

namespace mindspore {
class TestLogAdapter : public UT::Common {
 public:
  TestLogAdapter() {}

  void TearDown() override { (void)unsetenv(kMsExceptionDisplayLevel); }
};

/// Feature: Error message structure optimization, hierarchical printing
/// Description: set MS_EXCEPTION_DISPLAY_LEVEL to 0 and run pass
/// Expectation: dmsg and umsg both display
TEST_F(TestLogAdapter, TestParseExceptionMessage1) {
  (void)setenv(kMsExceptionDisplayLevel, "0", 1);
  try {
    MS_LOG(EXCEPTION) << "Exception Description#dmsg#The Dev Title1:#dmsg#The Dev Content1"
                      << "#umsg#The User Title1:#umsg#The User Content1";
  } catch (const std::exception &ex) {
    const std::string &exception_str = ex.what();
    MS_LOG(INFO) << exception_str;
    ASSERT_TRUE(exception_str.find("- C++ Call Stack: (For framework developers)") != std::string::npos);
  }
}

/// Feature: Error message structure optimization, hierarchical printing
/// Description: set MS_EXCEPTION_DISPLAY_LEVEL to 1 and run pass
/// Expectation: only umsg displays
TEST_F(TestLogAdapter, TestParseExceptionMessage2) {
  (void)setenv(kMsExceptionDisplayLevel, "1", 1);
  try {
    MS_EXCEPTION(TypeError) << "Exception Description#dmsg#The Dev Title1:#dmsg#The Dev Content1"
                            << "#umsg#The User Title1:#umsg#The User Content1";
  } catch (const std::exception &ex) {
    const std::string &exception_str = ex.what();
    MS_LOG(INFO) << exception_str;
    ASSERT_TRUE(exception_str.find("The Dev Content1") == std::string::npos);
  }
}

/// Feature: Framework exception info optimization
/// Description: Test that displays user note message when using internal expcetion
/// Expectation: User guide message displays
TEST_F(TestLogAdapter, TestInternalException) {
  const std::string title = "Framework Unexpected Exception Raised:";
  const std::string content =
    "This exception is caused by framework's unexpected error. Please create an issue at "
    "https://gitee.com/mindspore/mindspore/issues to get help.";

  // Test MS_INTERNAL_EXCEPTION(type)
  try {
    MS_INTERNAL_EXCEPTION(TypeError) << "test content1";
  } catch (const std::exception &ex) {
    const std::string &exception_str = ex.what();
    MS_LOG(INFO) << exception_str;
    ASSERT_TRUE(exception_str.find(title) != std::string::npos);
    ASSERT_TRUE(exception_str.find(content) != std::string::npos);
  }

  // Test MS_LOG(INTERNAL_EXCEPTION)
  try {
    MS_LOG(INTERNAL_EXCEPTION) << "test content2";
  } catch (const std::exception &ex) {
    const std::string &exception_str = ex.what();
    MS_LOG(INFO) << exception_str;
    ASSERT_TRUE(exception_str.find(title) != std::string::npos);
    ASSERT_TRUE(exception_str.find(content) != std::string::npos);
  }

  // Test MS_LOG(INTERNAL_EXCEPTION), too.
  try {
    MS_LOG(INTERNAL_EXCEPTION) << "test content3";
  } catch (const std::exception &ex) {
    const std::string &exception_str = ex.what();
    MS_LOG(INFO) << exception_str;
    ASSERT_TRUE(exception_str.find(title) != std::string::npos);
    ASSERT_TRUE(exception_str.find(content) != std::string::npos);
  }
}
}  // namespace mindspore
