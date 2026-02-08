/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "tools/error_handler/error_config.h"
#include "tools/error_handler/error_handler.h"
#include "utils/ms_utils.h"

namespace mindspore {
class TestErrorHandler : public UT::Common {
 public:
  TestErrorHandler() = default;
  virtual ~TestErrorHandler() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test UCE Exception instance.
/// Description:Test all interfaces in the instance.
/// Expectation: The return value of the normal interface is as expected, and the exception branch is expected to catch
/// the exception.
TEST_F(TestErrorHandler, test_interface) {
  auto &error_handler = tools::ErrorHandler::GetInstance();

  EXPECT_EQ(tools::TftConfig::GetInstance()->IsEnableUCE(), false);
  EXPECT_EQ(error_handler.HasThrownError(), false);
  EXPECT_EQ(error_handler.GetForceStopFlag(), false);
  EXPECT_EQ(error_handler.GetUceFlag(), false);
  EXPECT_EQ(error_handler.IsRebootNode(), false);
  EXPECT_EQ(error_handler.IsArf(), false);

  // test arf/uce/ttp basic interface  
  EXPECT_NO_THROW(error_handler.SetIsArf(true));
  EXPECT_NO_THROW(error_handler.SetRebootNode(true));
  EXPECT_EQ(error_handler.IsRebootNode(), true);
  EXPECT_EQ(error_handler.IsArf(), true);
  EXPECT_EQ(error_handler.HasThrownError(), true);

  EXPECT_NO_THROW(error_handler.SetForceStopFlag(true));
  EXPECT_EQ(error_handler.GetForceStopFlag(), true);

  EXPECT_NO_THROW(error_handler.SetRebootType("arf"));
  EXPECT_EQ(error_handler.GetRebootType(), "arf");

  EXPECT_EQ(error_handler.GetSuspectRemoteFlag(), false);
  EXPECT_EQ(error_handler.HasThrownError(), true);
}
}  // namespace mindspore
